import multiprocessing
import sys


def main():
    multiprocessing.freeze_support()
    if '--worker' in sys.argv:
        from papertranslator.offline import install_loopback_guard
        position=sys.argv.index('--worker')
        from pathlib import Path
        from papertranslator.storage import root,read_json
        folder=Path(sys.argv[position+1]).resolve()
        if folder.parent!=(root()/'jobs').resolve():raise ValueError('허용되지 않은 작업 경로')
        settings=read_json(folder/'state.json',{}).get('quality_policy',{})
        install_loopback_guard(allow_free_api=settings.get('mode')=='hybrid' and bool(settings.get('cloud',{}).get('enabled')))
        from papertranslator.jobs import run_job
        run_job(sys.argv[position+1], '--retry-failed' in sys.argv)
    else:
        from papertranslator.ui import main as ui_main
        return ui_main()


if __name__=='__main__':
    raise SystemExit(main())
