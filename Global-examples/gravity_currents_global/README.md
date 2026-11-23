# Global gravity currents

goSPL's optional marine gravity currents (hyperpycnal flows and turbidity currents) on
the global model of [`erodep_1My`](../erodep_1My) (a virtual planet at 251 Ma, 370,830
nodes, 1 Myr, marine deposition, global flexure), with dual lithology and stratigraphy
added. The run with gravity currents (`input-gravity.yml`) is compared with the same
model without them (`input-nogravity.yml`) and with them plus slope failure
(`input-failure.yml`).

1. Run `../erodep_1My/build_inputs.ipynb` (this example reuses its `vars_40` inputs).
2. `./run_all.sh` (three runs, 4 ranks each, about 5 minutes side by side).
3. Open `view_Results.ipynb`: gravity budget, global maps, one margin, filled
   stratigraphic sections (fine fraction and depositional facies) across it, and
   the effect of slope failure.

Needs a goSPL with the gravity-current module (the `environment.yml` of this branch
builds it). The regional example `Local-examples/gravity_currents` explains the module
in more detail.
