import multiprocessing
import sys


def main():
    multiprocessing.freeze_support()
    if '--worker' in sys.argv:
        from papertranslator.offline import install_loopback_guard
        install_loopback_guard()
        from papertranslator.jobs import run_job
        position=sys.argv.index('--worker')
        run_job(sys.argv[position+1], '--retry-failed' in sys.argv)
    else:
        from papertranslator.ui import main as ui_main
        return ui_main()


if __name__=='__main__':
    raise SystemExit(main())
