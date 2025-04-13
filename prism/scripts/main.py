import argparse
import logging

logging.basicConfig(level=logging.INFO)


def build_argparser():
    """
    Build the argument parser.
    """
    parser = argparse.ArgumentParser(
        description="PRISM: Platform for Research in Imaging and Signal Methodology"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser


def main():
    """
    Entry point of the script.
    """
    parser = build_argparser()
    args = parser.parse_args()
    
    logging.info("Running PRISM")
