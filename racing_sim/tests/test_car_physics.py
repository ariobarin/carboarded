import pymunk
from pymunk import Vec2d

from racing_sim.config.config import CarConfig
from racing_sim.physics.car import Car


def _make_car(config: CarConfig, speed: float = 0.0):
    space = pymunk.Space()
    car = Car(space, config, position=(0.0, 0.0))
    if speed:
        car.body.velocity = Vec2d(speed, 0.0)
    return space, car


def test_turning_slows_forward_speed():
    config = CarConfig(
        max_speed=2000.0,
        engine_power=0.0,
        accel_boost=0.0,
        lateral_friction=0.0,
        rolling_friction=0.0,
        air_drag=0.0,
        angular_damping=0.0,
        steering_power=500.0,
        steering_speed_ref=100.0,
        steering_min_factor=0.3,
    )

    _, car_turn = _make_car(config, speed=100.0)
    car_turn.apply_control(steering=1.0, throttle=0.0)
    car_turn.update_friction()
    turn_speed = car_turn.speed

    _, car_straight = _make_car(config, speed=100.0)
    car_straight.apply_control(steering=0.0, throttle=0.0)
    car_straight.update_friction()
    straight_speed = car_straight.speed

    assert turn_speed < straight_speed


def test_acceleration_is_stronger_at_low_speed():
    config = CarConfig(
        max_speed=1000.0,
        engine_power=500.0,
        accel_boost=0.6,
        lateral_friction=0.0,
        rolling_friction=0.0,
        air_drag=0.0,
        angular_damping=0.0,
        steering_power=0.0,
        steering_speed_ref=100.0,
        steering_min_factor=0.3,
    )

    low_space, low_car = _make_car(config, speed=0.0)
    low_car.apply_control(steering=0.0, throttle=1.0)
    low_car.update_friction()
    low_space.step(1.0 / 60.0)
    low_delta = low_car.speed

    high_space, high_car = _make_car(config, speed=800.0)
    high_car.apply_control(steering=0.0, throttle=1.0)
    high_car.update_friction()
    high_space.step(1.0 / 60.0)
    high_delta = high_car.speed - 800.0

    assert low_delta > high_delta


def test_steering_torque_reduces_with_speed():
    config = CarConfig(
        max_speed=2000.0,
        engine_power=0.0,
        accel_boost=0.0,
        lateral_friction=0.0,
        rolling_friction=0.0,
        air_drag=0.0,
        angular_damping=0.0,
        steering_power=500.0,
        steering_speed_ref=100.0,
        steering_min_factor=0.3,
    )

    _, slow_car = _make_car(config, speed=0.0)
    slow_car.apply_control(steering=1.0, throttle=0.0)
    slow_torque = abs(slow_car.body.torque)

    _, fast_car = _make_car(config, speed=200.0)
    fast_car.apply_control(steering=1.0, throttle=0.0)
    fast_torque = abs(fast_car.body.torque)

    assert slow_torque > fast_torque


def test_air_drag_scales_with_speed():
    config = CarConfig(
        max_speed=2000.0,
        engine_power=0.0,
        accel_boost=0.0,
        lateral_friction=0.0,
        rolling_friction=0.0,
        air_drag=0.02,
        angular_damping=0.0,
        steering_power=0.0,
        steering_speed_ref=100.0,
        steering_min_factor=0.3,
    )

    slow_space, slow_car = _make_car(config, speed=50.0)
    slow_car.update_friction()
    slow_space.step(1.0 / 60.0)
    slow_drop = 50.0 - slow_car.speed

    fast_space, fast_car = _make_car(config, speed=200.0)
    fast_car.update_friction()
    fast_space.step(1.0 / 60.0)
    fast_drop = 200.0 - fast_car.speed

    assert fast_drop > slow_drop


def test_throttle_response_smooths_initial_accel():
    config_fast = CarConfig(
        max_speed=2000.0,
        engine_power=600.0,
        accel_boost=0.0,
        throttle_response=1.0,
        lateral_friction=0.0,
        rolling_friction=0.0,
        air_drag=0.0,
        angular_damping=0.0,
        steering_power=0.0,
        steering_speed_ref=100.0,
        steering_min_factor=0.3,
    )
    fast_space, fast_car = _make_car(config_fast, speed=0.0)
    fast_car.apply_control(steering=0.0, throttle=1.0)
    fast_car.update_friction()
    fast_space.step(1.0 / 60.0)
    fast_delta = fast_car.speed

    config_slow = CarConfig(
        max_speed=2000.0,
        engine_power=600.0,
        accel_boost=0.0,
        throttle_response=0.2,
        lateral_friction=0.0,
        rolling_friction=0.0,
        air_drag=0.0,
        angular_damping=0.0,
        steering_power=0.0,
        steering_speed_ref=100.0,
        steering_min_factor=0.3,
    )
    slow_space, slow_car = _make_car(config_slow, speed=0.0)
    slow_car.apply_control(steering=0.0, throttle=1.0)
    slow_car.update_friction()
    slow_space.step(1.0 / 60.0)
    slow_delta = slow_car.speed

    assert slow_delta < fast_delta


def test_rolling_friction_scales_with_dt():
    config = CarConfig(
        max_speed=2000.0,
        engine_power=0.0,
        accel_boost=0.0,
        throttle_response=1.0,
        lateral_friction=0.0,
        rolling_friction=1.0,
        air_drag=0.0,
        angular_damping=0.0,
        steering_power=0.0,
        steering_speed_ref=100.0,
        steering_min_factor=0.3,
    )

    _, fast_dt_car = _make_car(config, speed=100.0)
    fast_dt_car.update_friction(dt=1.0 / 60.0)
    fast_drop = 100.0 - fast_dt_car.speed

    _, slow_dt_car = _make_car(config, speed=100.0)
    slow_dt_car.update_friction(dt=1.0 / 120.0)
    slow_drop = 100.0 - slow_dt_car.speed

    assert slow_drop < fast_drop


def test_lateral_velocity_is_removed_each_step():
    config = CarConfig(
        max_speed=2000.0,
        engine_power=0.0,
        accel_boost=0.0,
        throttle_response=1.0,
        lateral_friction=0.0,
        rolling_friction=0.0,
        air_drag=0.0,
        angular_damping=0.0,
        steering_power=0.0,
        steering_speed_ref=100.0,
        steering_min_factor=0.3,
        turning_drag=0.0,
    )

    _, car = _make_car(config)
    car.body.velocity = Vec2d(100.0, 50.0)

    car.update_friction(dt=1.0 / 60.0)

    assert abs(car.body.velocity.y) < 1e-6
