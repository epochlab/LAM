#!/usr/bin/env python3

def lookup_value(input, dict):
    return [v for k, v in dict.items() if input in k]