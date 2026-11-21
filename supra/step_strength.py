"""Apply resolved strengths in Supra's original conditional Euler loop."""


def scheduled_latent(backend, row, strengths):
    latent = row.latent
    schedule = row.schedule
    for time, delta, value in zip(schedule.timesteps, schedule.deltas, strengths, strict=True):
        velocity = backend.cfg(row, latent, latent.new_tensor(time), scale=value)
        latent = backend._advance(latent, velocity, delta)
    return latent
