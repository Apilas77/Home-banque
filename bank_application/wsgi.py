"""
WSGI config for bank_application project.

It exposes the WSGI callable as a module-level variable named ``calcul_default_risk``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'bank_application.settings')

application = get_wsgi_application()
