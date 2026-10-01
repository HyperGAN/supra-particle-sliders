"""Native CPU replay and game-only selection checks for bank trust."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from particlegan import E22Policy, RoutedRows, get_recipe, init
from supra.particle_game import ConditionalTokenCritic, update, patchify
from supra.particle_pilot import checkpoint, restore, state_digest
from scripts.experimental_e22_bank_game_trust import bank_game_trust, choose_fraction


@pytest.fixture(autouse=True)
def serial_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.autograd.set_multithreading_enabled(False):
            yield
    finally:
        torch.set_num_threads(previous)


class Encoder(nn.Module):
    def condition(self, context):
        return context


class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 4)
        self.second = nn.Linear(6, 4)
        self.register_buffer("log_mass", torch.zeros(128))


def native_loop():
    recipe = get_recipe("e22_routed", num_particles=128, z_dim=4, batch_size=4,
                        output_noise_std=.125)
    with torch.random.fork_rng(devices=[]):
        generator, router = nn.Linear(4, 16), Router()
        encoder = Encoder()
        critic = ConditionalTokenCritic(torch.ones(16), condition_dim=2, width=8, feature_dim=4)
        for module, role in ((generator, 0), (critic, 1), (encoder, 2), (router, 3)):
            init.deterministic_orthogonal_(module, seed=role)
        table = init.deterministic_orthogonal_(recipe.make_prior()).z.requires_grad_(True)
    optimizer = recipe.make_generator_optimizer([
        dict(params=list(generator.parameters()), lr=5e-5),
        dict(params=list(router.parameters()), lr=5e-5),
        dict(params=[table], lr=.0085)], latent_table=table, foreach=False)
    critic_optimizer = recipe.make_critic_optimizer(critic, ema_critic=deepcopy(critic), foreach=False)

    def forward(models, context, candidate, routing):
        first = routing.mix("first", (models["router"].first(context).unsqueeze(1) @ candidate.table.T) / 2.)
        query = models["router"].second(torch.cat((context, first[:, 0]), dim=-1)).unsqueeze(1)
        second = routing.mix("second", (query @ candidate.table.T) / 2.)
        patch = models["generator"]((first[:, 0] + second[:, 0]) / 2.)
        patches = patch[:, None].expand(-1, 256, -1)
        return patches.reshape(-1, 16, 16, 4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(-1, 4, 32, 32)

    def features(models, context, samples, targets):
        return models["critic"].features(patchify(samples - targets), context).flatten(1)

    spec = RoutedRows(model_forward=forward, features=features, sites=("first", "second"),
                      probe_interval=100, max_context_harm=0., output_error_guard=False)
    policy = E22Policy(recipe, generator, critic, table=table, encoder=encoder, router=router,
        generator_optimizer=optimizer, critic_optimizer=critic_optimizer,
        roles=[["generator", "router", "table"], ["critic"]], routed_rows=spec, seed=21)
    policy.attach_penalty(recipe.make_critic_penalty(critic_optimizer, collect_stats=True))
    context = torch.linspace(-.3, .4, 8).reshape(4, 2)
    return SimpleNamespace(policy=policy, data_rng=torch.Generator().manual_seed(7),
        paired_noise_rng=torch.Generator().manual_seed(43), config={"toy": "two-site-game-trust"},
        context=context, guard_context=context + .1, guard_targets=torch.zeros(4, 4, 32, 32))


def trial_update(loop):
    return update(loop, context=loop.context, target=torch.zeros(4, 4, 32, 32), batch_indices=torch.arange(4))


def test_game_guard_uses_both_training_games_and_prescribed_largest_fraction():
    baseline = dict(clean=1., noisy=2.)
    candidates = [(1., dict(clean=.9, noisy=2.1)), (.5, dict(clean=1.1, noisy=1.9)),
                  (.25, dict(clean=.99, noisy=2.)), (.125, dict(clean=.8, noisy=1.8))]
    assert choose_fraction(baseline, candidates) == .25
    assert choose_fraction(baseline, candidates[:2]) == 0.
    with pytest.raises(ValueError):
        choose_fraction(baseline, [(1., dict(clean=float("nan"), noisy=1.))])


def test_native_two_site_projection_preserves_moments_rngs_modes_and_exact_resume():
    loop = native_loop()
    initial = deepcopy(checkpoint(loop))
    native_rows = [trial_update(loop) for _ in range(2)]
    native_streams = state_digest(checkpoint(loop)["policy"]["streams"])
    restore(loop, initial)
    with bank_game_trust(loop) as helper:
        rows = []
        for _ in range(2):
            helper.begin_update()
            rows.append(trial_update(loop))
            record = helper.last_record
            assert record["original_native_Gpass_replayed_exact"]
            assert record["private_evaluations_state_unchanged"] and record["native_optimizer_moments_unchanged"]
            assert not record["output_metrics_used"]
            assert all(value <= 0 for value in record["accepted_new_minus_baseline"].values())
            assert record["dense_native_gradient_rows"] == 128
        records, final = deepcopy(helper.records), state_digest(checkpoint(loop))
    assert state_digest(checkpoint(loop)["policy"]["streams"]) == native_streams
    for row, native in zip(rows, native_rows):
        assert all(row[key] == native[key] for key in ("batch_indices", "base_noise_sums", "paired_rng_digest", "dv12_rng_digest"))
    restore(loop, initial)
    with bank_game_trust(loop) as helper:
        replay = []
        for _ in range(2):
            helper.begin_update()
            replay.append(trial_update(loop))
        assert replay == rows and helper.records == records
        assert state_digest(checkpoint(loop)) == final
        assert helper.state_dict()["completed_trials"] == 2
