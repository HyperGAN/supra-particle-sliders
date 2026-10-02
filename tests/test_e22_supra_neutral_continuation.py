"""Continuation software contracts, not full-Supra quality experiments.

The real native128×4, two-site CPU fixture resolves its backend with one update,
then uses an explicitly SYNTHETIC6400 clock and a separately advanced CPU7
sampling stream. Four actual updates verify the production recovery helper;
they cannot establish6400-update quality or actual GPU recovery.
"""
from copy import deepcopy
import types

import pytest
import torch

from scripts import experiment_e22_supra_neutral_initialization as held
from scripts.e22_supra_neutral_continuation_contract import (
    LOCAL_CHECKPOINTS, START, STOP, bind_code, code_identity, exact_recovery,
    held_manifest, nested_code, paired_row_contract, require_checkpoint_contract,
    ordinary_metadata_contract,
)
from supra.particle_game import update
from supra.particle_pilot import checkpoint, restore, state_digest
from test_experimental_e22_bank_game_trust import native_loop


def native_edit(loop):
    def require(name, condition):
        assert condition,name
    fn=bind_code(nested_code(held.main.__code__,"all_edit"),held.__dict__,
        dict(device=torch.device("cpu"),require=require,torch=torch,update=update))
    assert fn.__code__ is nested_code(held.main.__code__,"all_edit")
    return fn(loop)


@pytest.fixture
def synthetic_boundary(monkeypatch):
    import test_experimental_e22_bank_game_trust as fixture
    original=fixture.get_recipe
    monkeypatch.setattr(fixture,"get_recipe",lambda name,**kw:original(name,
        birth_death_backend="auto",reopen_guard="settled",**kw))
    threads=torch.get_num_threads()
    torch.set_num_threads(1)
    current=native_loop()
    current.fit_context=torch.linspace(-.3,.4,480).reshape(240,2)
    native_edit(current) # Real backend resolution, not a fabricated restore tag.
    current.policy.completed_steps=START # Explicit synthetic clock only.
    rng=torch.Generator().manual_seed(7)
    for _ in range(START): torch.randint(240,(4,),generator=rng)
    current.data_rng.set_state(rng.get_state())
    state=deepcopy(checkpoint(current))
    def fresh_restore():
        result=native_loop()
        result.fit_context=current.fit_context
        restore(result,state)
        assert state_digest(checkpoint(result))==state_digest(state)
        return result
    try:
        yield fresh_restore,state
    finally:
        torch.set_num_threads(threads)


def test_exact_production_recovery_uses_all_native_and_caller_owners(synthetic_boundary):
    factory,boundary=synthetic_boundary
    witness=exact_recovery(factory,native_edit,checkpoint,state_digest)
    assert witness["rows_exact"] and witness["state_exact"]
    assert witness["native_replay_updates"]==4
    assert [row["step"] for row in witness["rows"]]==[6401,6402]
    assert all(not row["hold"] and row["game_weight"]==1. and row["dense_gradient_rows"]==128
        for row in witness["rows"])
    expected=torch.Generator().set_state(boundary["data_rng"])
    assert [row["batch_indices"] for row in witness["rows"]]==[
        torch.randint(240,(4,),generator=expected).tolist() for _ in range(2)]
    # Replay's entire digest contains optimizers, EMA, routing/controller,
    # native noise/penalty/RNG, and the external data/paired-noise streams.
    assert set(boundary["policy"]["streams"])=={
        "latent_generator","penalty_generator","eval_generator","noise_generator"}
    assert len(boundary["policy"]["optimizers"])==2


@pytest.mark.parametrize("owner",["models","averages","optimizers","controller","streams","data_rng","paired_noise_rng"])
def test_recovery_rejects_any_changed_final_owner(owner):
    counter=0
    def factory():
        nonlocal counter
        counter+=1
        return dict(value=0,copy=counter)
    def update_small(loop):
        loop["value"]+=1
        return dict(step=START+loop["value"],hold=False,game_weight=1.)
    def state(loop):
        value={key:0 for key in ("models","averages","optimizers","controller","streams","data_rng","paired_noise_rng")}
        value[owner]=loop["copy"]
        return value
    with pytest.raises(ValueError,match="not exact"):
        exact_recovery(factory,update_small,state,state_digest)


def test_bound_code_globals_and_closures_are_isolated_from_held_module():
    def code_factory(offset):
        def fn(value):return value+offset+TOKEN
        return fn.__code__
    code=code_factory(1)
    namespace={"TOKEN":3}
    fn=bind_code(code,namespace,dict(offset=2))
    assert fn(4)==9 and fn.__code__ is code
    fn.__globals__["TOKEN"]=9
    assert namespace["TOKEN"]==3
    with pytest.raises(ValueError,match="missing explicit"):
        bind_code(code,namespace,{})
    assert code_identity(code)==code_identity(fn.__code__)
    changed=code.replace(co_consts=tuple(1 if item is None else item for item in code.co_consts))
    assert code_identity(changed)!=code_identity(code)
    manifest=held_manifest(held)
    assert manifest["helpers"]["all_edit"]["closure_names"]==["device","require","torch","update"]


def config_state():
    config=dict(training_schedule="fresh_editing_only_v1",preservation_game_weight=0.,
        output_error_guard=False,max_feature_context_harm=0.,recipe=dict(total_steps=None,betas=(0.,.999)),
        particle_init="sampled_hb_neutral_v1")
    return dict(config=config,policy=dict(completed_steps=START,recipe=deepcopy(config["recipe"])))


def test_json_contract_comparison_preserves_raw_tuple_config_for_public_restore():
    import json
    state=config_state();before=deepcopy(state)
    expected=json.loads(json.dumps(state["config"]))
    require_checkpoint_contract(state,expected,START)
    assert state==before and isinstance(state["config"]["recipe"]["betas"],tuple)


def test_actual_historical_json_encoded_metadata_is_decoded_before_pin_comparison():
    import json
    host=dict(model_id="SupraLabs/Supra2-IMG",model_revision="model",text_encoder_revision="text",vae_revision="vae")
    values=dict(host,step=12800,rank=16,alpha=16,prompts_sha256="prompts",
        targets=["ctx_proj","cross_attn.q","cross_attn.kv","cross_attn.proj","self_attn.qkv","self_attn.proj"])
    metadata={key:json.dumps(value) for key,value in values.items()}
    metadata["format"]=json.dumps("supra-native-lora-v1")
    ordinary_metadata_contract(metadata,host,12800,"prompts")
    for key,value in (("model_revision","changed"),("step",6400),("rank",8),("prompts_sha256","changed")):
        altered=dict(metadata,**{key:json.dumps(value)})
        with pytest.raises(ValueError,match="pins changed"):
            ordinary_metadata_contract(altered,host,12800,"prompts")


@pytest.mark.parametrize("key,value",[("training_schedule","mixed"),("preservation_game_weight",.1),
    ("output_error_guard",True),("max_feature_context_harm",1e-4)])
def test_continuation_refuses_game_or_sampling_interventions(key,value):
    state=config_state()
    state["config"][key]=value
    from scripts.e22_supra_neutral_continuation_contract import canonical
    with pytest.raises(ValueError,match="native editing-only"):
        require_checkpoint_contract(state,canonical(state["config"]),START)


def test_both_endpoints_and_all_new_checkpoint_clocks_are_fixed():
    assert LOCAL_CHECKPOINTS==(6402,*range(6800,12801,400)) and len(LOCAL_CHECKPOINTS)==17
    assert START==6400 and STOP==12800
    row=dict(step=6401,hold=False,game_weight=1.,batch_indices=[0,1,2,3],base_noise_sums=[1.,2.],paired_rng_digest="same")
    paired_row_contract(row,deepcopy(row),6401,[0,1,2,3])
    for key,value in (("hold",True),("game_weight",.1),("paired_rng_digest","drift"),("batch_indices",[3,2,1,0])):
        changed=dict(row,**{key:value})
        with pytest.raises(ValueError):paired_row_contract(row,changed,6401,[0,1,2,3])


def test_independent_reduction_rejects_false_gain_or_missing_subject():
    from scripts.review_e22_supra_neutral_initialization import Reviewer,check_evaluation,COUNTS
    data,result={},{}
    for pool,count in COUNTS.items():
        context=torch.zeros(count,4099)
        context[:,4097]=torch.tensor([4,5,15,16,17,18]*(count//6)+[4]*(count%6))
        data[pool]=dict(context=context)
        records=[dict(index=i,source_caption_id=int(context[i,4097]),time=0.,mse_diagnostic=1.,D1856=2.,D6400=3.) for i in range(count)]
        result[pool]=dict(count=count,records=records,rmse_diagnostic=1.,D1856=2.,D6400=3.)
    check_evaluation(Reviewer(),result,data,"small")
    altered=deepcopy(result);altered["test"]["D1856"]=1.
    with pytest.raises(ValueError,match="game reduction"):check_evaluation(Reviewer(),altered,data,"bad")
    altered=deepcopy(result);altered["test"]["records"].pop()
    with pytest.raises(ValueError,match="count"):check_evaluation(Reviewer(),altered,data,"bad")


@pytest.mark.parametrize("tamper",["tensor","served_source"])
def test_independent_export_review_rejects_actual_tensor_or_head_metadata_change(tmp_path,tamper):
    import json
    from safetensors.torch import save_file
    from scripts.review_e22_supra_neutral_initialization import Reviewer,review_export
    generator,router,flags={},dict(log_mass=torch.zeros(128)),{}
    sites=["site"+str(i) for i in range(71)]
    for i in range(71):
        for suffix,shape in (("down.weight",(16,1)),("up.weight",(1,16)),
                             ("bridge.weight",(16,20)),("bridge.bias",(16,))):
            name=sites[i]+"."+suffix
            generator[name]=torch.ones(shape);flags[name]=True
        router[f"site_queries.query_{i:03d}.weight"]=torch.ones(4,1)
        router[f"site_queries.query_{i:03d}.bias"]=torch.ones(4)
    policy=dict(models=dict(generator=generator,router=router),requires_grad=dict(generator=flags),table=torch.ones(128,4))
    state=dict(policy=policy,config=dict(sites=sites))
    tensors={"generator."+name:value for name,value in generator.items()}
    tensors.update({"router."+name:value for name,value in router.items() if name!="log_mass"})
    tensors.update({"bank.table":policy["table"],"bank.log_mass":router["log_mass"]})
    config=dict(architecture="gated_particle_v3",served_source="fast",completed_steps=9600,
        extra=dict(particle_init="sampled_hb_neutral_v1",training_schedule="fresh_editing_only_v1"),
        rank=16,z_dim=4,num_particles=128,cfg=3,sampling="clean",routed_geometry="mass_atoms_v1",
        sites=sites,site_input_dims={name:1 for name in sites})
    metadata=dict(format="supra_particlegan_clean_v3",config=json.dumps(config),
        native_pins=json.dumps(dict(backend_sha256="fixed")))
    path=tmp_path / "adapter.safetensors"
    save_file(tensors,str(path),metadata=metadata)
    review_export(Reviewer(),path,state,"sampled_hb_neutral",9600,"fixed")
    if tamper=="tensor":
        tensors={name:value.clone() for name,value in tensors.items()}
        tensors["bank.table"][0,0]+=1
    else:
        config["served_source"]="averaged"
        metadata["config"]=json.dumps(config)
    save_file(tensors,str(path),metadata=metadata)
    with pytest.raises(ValueError):review_export(Reviewer(),path,state,"sampled_hb_neutral",9600,"fixed")
