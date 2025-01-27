#!/bin/bash
# Run the integration tests.
# Be sure to install the test requirements first!
#    pip install -r requirements.txt

# Run the tests
coverage run -m pytest test_*.py 