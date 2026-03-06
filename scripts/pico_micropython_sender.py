# MicroPython sketch for Raspberry Pi Pico (run on Pico with Thonny)
# Streams CSV: t,ch0,ch1,...,chN
import machine
import utime

ADC_PINS = [26, 27, 28]  # adjust to your wiring
SR_HZ = 1000

adcs = [machine.ADC(pin) for pin in ADC_PINS]
t0 = utime.ticks_ms()
period_us = int(1_000_000 / SR_HZ)

while True:
    ts = utime.ticks_diff(utime.ticks_ms(), t0) / 1000.0
    vals = [adc.read_u16() for adc in adcs]
    print(",".join([str(ts)] + [str(v) for v in vals]))
    utime.sleep_us(period_us)
