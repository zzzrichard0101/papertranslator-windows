from papertranslator.__main__ import main

if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        from papertranslator.storage import root
        (root()/'startup-error.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise
