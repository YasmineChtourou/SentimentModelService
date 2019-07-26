#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import argparse
from chatbot_project import config


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot_project.settings')
    argv = sys.argv
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('env',choices = ['dev', 'prod'])
    args, argv = parser.parse_known_args()
    if args.env=='dev':
           config.repo = os.path.abspath('')+'/'  
    if args.env=='prod':
           config.repo = os.path.abspath('')+'/sentimentmodelservice/'
    #from django.conf import settings
    #settings.env= args.env
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(argv)


if __name__ == '__main__':
    main()
