"""Restrict Python worker network calls without changing Windows firewall settings."""
import ipaddress
import sys


def install_loopback_guard():
    def audit(event, args):
        if event in {'socket.connect', 'socket.connect_ex'}:
            address=args[1]
            if isinstance(address, tuple):
                try: local=ipaddress.ip_address(address[0]).is_loopback
                except ValueError: local=False
                if not local: raise PermissionError('Worker permits only numeric loopback network addresses')
        if event=='socket.getaddrinfo':
            host=args[0]
            try: local=ipaddress.ip_address(host).is_loopback
            except ValueError: local=False
            if not local: raise PermissionError('External DNS is disabled in translation worker')
    sys.addaudithook(audit)
