#!/usr/bin/env python3
"""
Environment Variable Checker

This script helps you check if your API keys and other environment variables are set correctly.
"""

import os

def check_env_vars():
    """Check important environment variables."""
    vars_to_check = [
        'OPENAI_API_KEY',
        'PYTHONPATH',
        'PATH'
    ]

    print("🔍 Environment Variable Check")
    print("=" * 40)

    for var in vars_to_check:
        value = os.getenv(var)
        if value:
            if 'KEY' in var.upper() or 'SECRET' in var.upper():
                # Don't print actual key values for security
                print(f"✅ {var}: Set (length: {len(value)})")
            else:
                print(f"✅ {var}: {value[:50]}..." if len(value) > 50 else f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")

    print("\n💡 To set OPENAI_API_KEY permanently:")
    print("   Run in PowerShell as Administrator:")
    print("   [Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'your-key-here', 'User')")

if __name__ == "__main__":
    check_env_vars()