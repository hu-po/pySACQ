# Action is two floats [main engine, left-right engines].
# Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
# Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off


# Observation space, according to source:
# state = [
#     (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
#     (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_W / SCALE / 2),
#     vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
#     vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
#     self.lander.angle,
#     20.0 * self.lander.angularVelocity / FPS,
#     1.0 if self.legs[0].ground_contact else 0.0,
#     1.0 if self.legs[1].ground_contact else 0.0
# ]

#
# Auxiliary Rewards:
# Touch/No Touch. Maximizing whether legs are touching or not.
# Hover. Minimize the planar movement of the lander craft
# Upright. Minimize the lander angle.
# Controlled. Minimize the rotational movement of ht lander craft
# Close. Distance between lander craft and base is below some threshold
#
# Extrinsic Rewards:
# Success: Is the lander able to land successfully
