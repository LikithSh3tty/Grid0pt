"""Evaluation harness for the Grid0pt paper (design note section 11).

Three pieces, deliberately separated so a result can be traced back to what
produced it:

  corpus   -- the instances, generated deterministically from code rather than
              shipped as data, so the corpus IS the generator and a reader
              reproduces it by running it.
  methods  -- every solver under test behind one interface: the baselines the
              paper measures against, the Grid0pt pipeline, and the ablations
              that switch one component off at a time.
  run      -- the driver: methods x corpus, one row per pair, written as CSV
              and JSON with the metrics section 11 asks for.

Run it with `python -m evaluation.run` from the backend directory.
"""
