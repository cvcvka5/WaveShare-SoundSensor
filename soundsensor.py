from machine import Pin
import _thread
import time

__author__ = "cvcvka5"
__version__ = "1.0.6"


class SoundSensor(object):
    def __init__(self, sensor_pin, reverse_threshold = False):
        self._mic = Pin(sensor_pin, Pin.IN)
        self._clap_count = 0
        self.reverse_threshold = reverse_threshold
        
        
    def _clap_counter_monothread(self, callback, lower_wait_threshold_ms, upper_wait_threshold_ms):
        # Last clap time in milliseconds.
        last = 0
        clap = 0
        # Run callback after a certain amount of milliseconds.
        exec_after_ms = 500
        
        while True:
            # Runs the callback if the last clap time passes certain milliseconds.
            if time.ticks_ms() - last > exec_after_ms and clap > 0:
                callback(clap)
                clap = 0
            
            # Some sound sensors have a bug where the sound output must be reversed.
            val = not self._mic.value() if self.reverse_threshold else self._mic.value()
            if val:
                t = time.ticks_ms()
                
                # Clap + 1 if the consecutive clap comes after 150ms and before 350ms.
                if lower_wait_threshold_ms < t-last < upper_wait_threshold_ms:
                    clap += 1
                # If the the clap comes after 350ms but not before 150ms then render it as a first-time clap.
                if t-last >= upper_wait_threshold_ms:
                    clap = 1
                
                last = t
    
    def clap_counter(self, callback, lower_wait_threshold_ms = 150, upper_wait_threshold_ms = 350):
        _thread.start_new_thread(self._clap_counter_monothread, [callback, lower_wait_threshold_ms, upper_wait_threshold_ms])

