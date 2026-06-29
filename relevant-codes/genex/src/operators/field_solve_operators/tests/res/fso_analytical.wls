#!/usr/bin/env wolframscript
(*
  This script calculates sources for the field solver unit tests.
  The equilibrium must be provided as an input, current valid options are slab,
  circular and salpha.
  Files are written automatically to the output directory provided.
*)

(* Handle command line arguments and check input *)
argv = Rest @ $ScriptCommandLine;
argc = Length @ argv;

If[argc != 1 && argc != 2,
   Throw["usage: wolframscript fso_analytical.wls [OUTPUT_DIR] EQUILIBRIUM"]]

equilibrium = argv[[argc]];

If[equilibrium != "slab" && equilibrium != "circular" \
   && equilibrium != "salpha",
   Throw["error: equilibrium " <> equilibrium <> " not valid! " \
         <> "use slab, circular or salpha."]]

If[argc == 1, path = "./", path = argv[[1]]]
If[!DirectoryQ[path], Throw["error: path " <> path <> " does not exist!"]]
If[StringTake[path, -1]!="/", path = path <> "/"]

(* Include Packages *)

CWD = Directory[]
SetDirectory["../../../../test_utils/mathematica_packages/"]
(* Import the F90Format function *)
<< Fortran90`
SetDirectory[CWD]

(* Start FSO *)
Print["Welcome to FSO analytical, received equilibrium type " <> equilibrium]

(* General constants *)
masses := {1, 0.5}

(* Definition of the coordinate system and magnetic field *)

coordinatesystem = If[equilibrium == "salpha", "Cylindrical", "Cartesian"]

rho[R_, Z_] := Switch[equilibrium, "slab", R, \
    "circular", CoordinateTransform["Cartesian" -> "Polar", {R, Z}][[1]], \
    "salpha", CoordinateTransform["Cartesian" -> "Polar", \
                                  {R - 1.0, Z}][[1]] / minorr]

theta[R_, Z_] := Switch[equilibrium, "slab", Z, \
    "circular", CoordinateTransform["Cartesian"-> "Polar", {R, Z}][[2]], \
    "salpha", CoordinateTransform["Cartesian"-> "Polar", {R - 1.0, Z}][[2]]]

jacobian[R_] := Switch[equilibrium, "slab", 1, "circular", 1, "salpha", R]

(* Definition of the test functions *)

testfunc[R_, phi_, Z_] :=
    Sin[2 * nrad * Pi * (rho[R, Z] - rhomin) / (rhomax - rhomin)] \
    * Sin[2 * npol * theta[R, Z]]

densi[R_, phi_ ,Z_] :=
    0.5 + Sin[nrad * Pi * (rho[R, Z] - rhomin) / (rhomax - rhomin)]^2 \
        * Sin[npol * theta[R, Z]]^2

dense[R_, phi_, Z_] :=
    0.5 + Cos[nrad * Pi * (rho[R, Z] - rhomin) / (rhomax - rhomin)]^2 \
        * Cos[npol * theta[R, Z]]^2

massdens[R_, phi_, Z_] :=
    masses[[1]] * densi[R, phi, Z] + masses[[2]] * dense[R, phi, Z]

lambdaohmslaw[R_, phi_, Z_] := densi[R, phi, Z] + dense[R, phi, Z]

coqneq[R_, phi_, Z_] := jacobian[R] * (rhoref / Lref)^2 * massdens[R, phi, Z]

(* Evaluation of the field equations with the test functions *)

bqneq[R_, phi_, Z_] := -1 / jacobian[R] \
    * Div[coqneq[R, phi, Z] * Grad[testfunc[R, phi, Z], {R, Z}, "Cartesian"], \
          {R, Z}, "Cartesian"]

bampslaw[R_, phi_, Z_] := -1 / jacobian[R] \
    * Div[jacobian[R] * (rhoref / Lref)^2 \
          * Grad[testfunc[R, phi, Z], {R, Z}, "Cartesian"], \
          {R, Z}, "Cartesian"]

bbpareq[R_, phi_, Z_] := testfunc[R, phi, Z]

bohmslaw[R_, phi_, Z_] := -1 / jacobian[R] \
    * Div[jacobian[R] * (rhoref / Lref)^2 \
          * Grad[testfunc[R, phi, Z], {R, Z}, "Cartesian"], \
          {R, Z}, "Cartesian"] \
    + lambdaohmslaw[R, phi, Z] * testfunc[R, phi, Z]

(* Write the result to file *)
Print["   writing to file in path ", path]

Export[StringJoin[path, "test_func_", equilibrium, ".txt"],
    {"test_func = " <> F90Format[testfunc[R, phi, Z]]}];
Export[StringJoin[path, "b_qn_eq_", equilibrium, ".txt"],
    {"b_qn_eq = " <> F90Format[bqneq[R, phi, Z]]}];
Export[StringJoin[path, "co_qn_eq_", equilibrium, ".txt"],
    {"co_qn_eq = " <> F90Format[coqneq[R, phi, Z]]}];
Export[StringJoin[path, "b_amps_law_", equilibrium, ".txt"],
    {"b_amps_law = " <> F90Format[bampslaw[R, phi, Z]]}];
Export[StringJoin[path, "b_bpar_eq_", equilibrium, ".txt"],
    {"b_bpar_eq = " <> F90Format[bbpareq[R, phi, Z]]}];
Export[StringJoin[path, "b_ohms_law_", equilibrium, ".txt"],
    {"b_ohms_law = " <> F90Format[bohmslaw[R, phi, Z]]}];
Export[StringJoin[path, "lambda_ohms_law_", equilibrium, ".txt"],
    {"lambda_ohms_law = " <> F90Format[lambdaohmslaw[R, phi, Z]]}];
