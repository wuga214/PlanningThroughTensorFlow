import tensorflow as tf

hvac_settings = {
    "cap": tf.constant(80.0, dtype=tf.float32),
    "outside_resist": tf.constant(4.0, dtype=tf.float32),
    "hall_resist": tf.constant(2.0, dtype=tf.float32),
    "wall_resist": tf.constant(1.5, dtype=tf.float32),
    "cap_air": tf.constant(1.006, dtype=tf.float32),
    "cost_air": tf.constant(1.0, dtype=tf.float32),
    "time_delta": tf.constant(1.0, dtype=tf.float32),
    "temp_air": tf.constant(40.0, dtype=tf.float32),
    "temp_up": tf.constant(23.5, dtype=tf.float32),
    "temp_low": tf.constant(20.0, dtype=tf.float32),
    "temp_outside": tf.constant(6.0, dtype=tf.float32),
    "temp_hall": tf.constant(10.0, dtype=tf.float32),
    "penalty": tf.constant(20000.0, dtype=tf.float32),
    "air_max": tf.constant(10.0, dtype=tf.float32)
}


