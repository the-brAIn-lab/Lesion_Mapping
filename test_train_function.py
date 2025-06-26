#!/usr/bin/env python3
"""
Test the actual train_model function
"""
import sys
import traceback
sys.path.append('.')

try:
    print("Testing train_model function...")
    from correct_full_training import train_model
    
    print("✅ train_model imported successfully")
    print("Calling train_model()...")
    
    result = train_model()
    print(f"train_model result: {result}")
    
except Exception as e:
    print(f"❌ train_model failed: {e}")
    traceback.print_exc()
