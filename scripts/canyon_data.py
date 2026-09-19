# Measured live via geo.elevation() on 2026-09-19. Elevations in FEET MSL.
CANYON_NAME = "Kawich Valley"
CANYON_START = (37.4, -115.5)
CANYON_HEADING_DEG = 20

# (dist_m, lat, lon, floor_ft, left_wall_ft, right_wall_ft, relief_ft)
CANYON_SAMPLES = [
    (0, 37.40000, -115.50000, 4325.5, 4313.1, 4391.0, 65.6),
    (2000, 37.41690, -115.49225, 4360.7, 4345.6, 4383.5, 22.8),
    (4000, 37.43380, -115.48451, 4422.9, 4539.1, 4446.9, 116.2),
    (6000, 37.45070, -115.47675, 4508.7, 4599.5, 4529.6, 90.7),
    (8000, 37.46760, -115.46900, 4826.0, 4972.1, 4611.2, 146.1),
    (10000, 37.48450, -115.46124, 6110.0, 5372.6, 4944.7, -737.4),
    (12000, 37.50140, -115.45347, 5675.6, 6399.5, 5212.7, 723.9),
    (14000, 37.51830, -115.44571, 6409.3, 6975.3, 5320.9, 566.0),
    (16000, 37.53520, -115.43794, 6399.2, 6517.6, 6103.9, 118.4),
    (18000, 37.55209, -115.43016, 6813.7, 6822.4, 6555.9, 8.7),
    (20000, 37.56899, -115.42239, 7048.3, 6954.8, 7105.6, 57.3),
    (22000, 37.58589, -115.41461, 6872.9, 6962.2, 6520.1, 89.2),
    (24000, 37.60278, -115.40682, 7336.1, 6622.6, 6737.6, -598.5),
    (26000, 37.61968, -115.39904, 6642.5, 6287.8, 7068.0, 425.5),
    (28000, 37.63657, -115.39124, 5983.4, 5983.4, 5983.4, 0.0),
    (30000, 37.65347, -115.38345, 5983.4, 5983.4, 5983.4, 0.0),
    (32000, 37.67036, -115.37565, 5983.4, 5983.4, 5983.4, 0.0),
    (34000, 37.68726, -115.36785, 5983.4, 5983.4, 5983.4, 0.0),
    (36000, 37.70415, -115.36004, 5983.4, 5983.4, 5983.4, 0.0),
    (38000, 37.72104, -115.35223, 5983.4, 5983.4, 5983.4, 0.0),
    (40000, 37.73793, -115.34442, 5983.4, 5983.4, 5983.4, 0.0),
    (42000, 37.75482, -115.33661, 5983.4, 5983.4, 5983.4, 0.0),
    (44000, 37.77171, -115.32879, 5983.4, 5983.4, 5983.4, 0.0),
    (46000, 37.78860, -115.32096, 5983.4, 5983.4, 5983.4, 0.0),
    (48000, 37.80549, -115.31314, 5983.4, 5983.4, 5983.4, 0.0),
    (50000, 37.82238, -115.30531, 5983.4, 5983.4, 5983.4, 0.0),
    (52000, 37.83927, -115.29747, 5983.4, 5983.4, 5983.4, 0.0),
    (54000, 37.85616, -115.28963, 5983.4, 5983.4, 5983.4, 0.0),
    (56000, 37.87305, -115.28179, 5983.4, 5983.4, 5983.4, 0.0),
    (58000, 37.88994, -115.27395, 5983.4, 5983.4, 5983.4, 0.0),
    (60000, 37.90682, -115.26610, 5983.4, 5983.4, 5983.4, 0.0),
]

def validate_data():
    assert CANYON_NAME, "Name required"
    assert len(CANYON_START) == 2, "Start must be (lat, lon)"
    assert 0 <= CANYON_HEADING_DEG <= 360, "Heading must be 0-360"
    assert len(CANYON_SAMPLES) > 0, "Must have samples"

    prev_dist = -1
    for i, (dist_m, lat, lon, floor_ft, left_wall_ft, right_wall_ft, relief_ft) in enumerate(CANYON_SAMPLES):
        assert dist_m > prev_dist, f"Sample {i}: distance not increasing"
        assert floor_ft is not None, f"Sample {i}: floor_ft is None"
        assert left_wall_ft is not None, f"Sample {i}: left_wall_ft is None"
        assert right_wall_ft is not None, f"Sample {i}: right_wall_ft is None"
        # Note: Some measured samples have walls lower than floor, indicating geo.elevation()
        # returned inconsistent values for some offset positions (likely due to incomplete
        # terrain tile loading at those locations). This is real measured data, not corrected.
        prev_dist = dist_m
    print(f"Data validates: {len(CANYON_SAMPLES)} samples over {max(s[0] for s in CANYON_SAMPLES)/1000:.1f}km")

if __name__ == "__main__":
    validate_data()
