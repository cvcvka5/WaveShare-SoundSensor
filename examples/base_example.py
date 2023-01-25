# Base example where the sound sensor prints out your clap count.

from soundsensor import SoundSensor
from machine import Pin

# NOTE: Remove this line if you're using the 3V3 on your Pico.
# SoundSensor VCC Pin On.
Pin(13, Pin.OUT).on()

# SoundSensor OUT Pin.
sound_out_pin = 14

# 'reverse_threshold' reverses the output of the sound_sensor by doing 'not output.value()' instead of just using the 'output.value()'.
# This is required since some Sound Sensors have a bug where the threshold comes out as '0' when it detects a sound.
sound = SoundSensor(sound_out_pin, reverse_threshold=False)

# Tinker with the 'lower_wait_threshold_ms' and the 'upper_wait_threshold_ms' based on what suits you the best.
sound.clap_counter(lambda clap_count: print(clap_count), lower_wait_threshold_ms = 150, upper_wait_threshold_ms = 350)
