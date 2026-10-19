"""Restrict Python worker network calls without changing Windows firewall settings."""
import ipaddress
import sys
import socket


def install_loopback_guard(allow_free_api=False):
    approved=set()
    original_resolver=socket.getaddrinfo
    def hostname(host):return host.decode('ascii') if isinstance(host,bytes) else str(host)
    def loopback(host):
        try:return ipaddress.ip_address(hostname(host)).is_loopback
        except ValueError:return False
    def resolve(host,port,*args,**kwargs):
        api=allow_free_api and hostname(host)=='openrouter.ai' and str(port)=='443'
        if not api and not loopback(host):raise PermissionError('External DNS is disabled in translation worker')
        results=original_resolver(host,port,*args,**kwargs)
        if api:
            addresses={row[4][0] for row in results}
            if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
                raise PermissionError('API DNS returned a non-public address')
            approved.update(addresses)
        return results
    def audit(event, args):
        if event in {'socket.connect', 'socket.connect_ex'}:
            address=args[1]
            if isinstance(address, tuple):
                try: local=ipaddress.ip_address(address[0]).is_loopback
                except ValueError: local=False
                api=allow_free_api and address[0] in approved and address[1]==443
                if not local and not api: raise PermissionError('Worker permits only loopback or explicitly enabled free API addresses')
        if event=='socket.getaddrinfo':
            host=args[0]
            local=loopback(host)
            api=allow_free_api and hostname(host)=='openrouter.ai' and str(args[1])=='443'
            if not local and not api: raise PermissionError('External DNS is disabled in translation worker')
    sys.addaudithook(audit)
    if allow_free_api:socket.getaddrinfo=resolve
