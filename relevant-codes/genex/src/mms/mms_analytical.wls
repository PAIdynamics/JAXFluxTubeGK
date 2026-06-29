#!/usr/bin/env wolframscript
(* This script calculates solutions and sources for the MMS tests.
   The equilibrium must be provided as an input. Current valid options are slab,
   circular, salpha, and dommaschk.
   Files are written automatically to the output directory provided. *)

(* Handle command line arguments and check input *)
argv = Rest @ $ScriptCommandLine;
argc = Length @ argv;

If[argc != 1 && argc != 2,
   Throw["usage: wolframscript mms_analytical.wls [OUTPUT_DIR] EQUILIBRIUM"]]

equilibrium = argv[[argc]];

If[equilibrium != "slab" && equilibrium != "circular" \
   && equilibrium != "salpha" && equilibrium != "dommaschk",
   Throw["error: equilibrium " <> equilibrium <> " not valid! " \
         <> "use slab, circular, salpha, or dommaschk."]]

If[argc == 1, path = "./" <> equilibrium, path = argv[[1]]]
If[!DirectoryQ[path], Throw["error: path " <> path <> " does not exist!"]]
If[StringTake[path, -1]!="/", path = path <> "/"]

ElapsedTime := \
    "(t =" <> \
        StringPadLeft[ToString[NumberForm[TimeUsed[] - t1, {7, 3}]], 8] <> \
    " s) "

(* Include packages. By using a relative path, this step assumes that the user
   is running the script directly in its local directory. *)

CWD = Directory[]
SetDirectory["../test_utils/mathematica_packages"]
(* Import the F90Format function *)
<< Fortran90`
(* Import the absBDommaschk and BDommaschk functions *)
<< DommaschkEquilibrium`
SetDirectory[CWD]

(* Start MMS *)
t1 = TimeUsed[]
Print[ElapsedTime <> "Welcome to MMS analytical, received equilibrium type "
                  <> equilibrium]

(* General constants *)
masses := {1, 1}
charges := {1, -1}
(* NOTE: tempScalings can be used to test the MMS with
         different normalizations in the future *)
tempScalings := {1, 1}
T := 1

(* Definition of the coordinate system and magnetic field *)

coordinatesystem = If[equilibrium == "salpha" ||
                      equilibrium == "dommaschk", "Cylindrical", "Cartesian"]

(* For Dommaschk, a rotated ellipse which approximates the flux surfaces is used
   for rho. The parameters of the ellipse (location of the ellipse center,
   toroidal mode number of the rotation, and major and minor axes) are optimized
   so that the boundary flux surfaces have as constant a rho as possible. *)
rho[R_, phi_, Z_] := Switch[equilibrium, "slab", R, \
    "circular", CoordinateTransform["Cartesian" -> "Polar", {R, Z}][[1]], \
    "salpha", CoordinateTransform["Cartesian" -> "Polar", \
                                  {R - 1.0, Z}][[1]] / minorr, \
    "dommaschk", Sqrt[ ( (R - 0.999) * Cos[2.5 * phi] \
                       + (Z        ) * Sin[2.5 * phi])^2 / ellax1^2\
                     + ( (R - 0.999) * Sin[2.5 * phi] \
                       - (Z        ) * Cos[2.5 * phi])^2 / ellax2^2]]

theta[R_, Z_] := Switch[equilibrium, "slab", Pi * Z, \
    "circular", CoordinateTransform["Cartesian"-> "Polar", {R, Z}][[2]], \
    "salpha", CoordinateTransform["Cartesian"-> "Polar", {R - 1.0, Z}][[2]], \
    "dommaschk", CoordinateTransform["Cartesian"-> "Polar", {R - 1.0, Z}][[2]]]

jacobian[R_] := Switch[equilibrium, "slab", 1, "circular", 1, "salpha", R, \
                                    "dommaschk", R]

qsalpha[R_, Z_] := q + shear * rho[R, 0.0, Z]^2

absBfunc[R_, phi_, Z_] := Switch[equilibrium, "slab", 1, \
    "circular", Sqrt[1 + rho[R, phi, Z]^2 / q^2], \
    "salpha", Sqrt[1 + ((R - 1)^2 + Z^2) / qsalpha[R, Z]^2] / R, \
    "dommaschk", absBDommaschk[R, phi, Z]]

(* Magnetic field unit vector in cylindrical coordinates (R, phi, Z) *)
b[R_, phi_, Z_] := Switch[equilibrium, "slab", {0, 1, 0}, \
    "circular", {-Z / Sqrt[R^2 + Z^2 + q^2], q / Sqrt[R^2 + Z^2 + q^2], \
                 R / Sqrt[R^2 + Z^2 + q^2]}, \
    "salpha", {-Z / Sqrt[(R - 1)^2 + Z^2 + qsalpha[R, Z]^2], \
               qsalpha[R,Z] / Sqrt[(R - 1)^2 + Z^2 + qsalpha[R, Z]^2], \
               (R - 1) / Sqrt[(R - 1)^2 + Z^2 + qsalpha[R, Z]^2]}, \
    "dommaschk", BDommaschk[R, phi, Z] / absBfunc[R, phi, Z]]

BstarES[R_, phi_, Z_, vp_, sigma_] :=
    Simplify[absBfunc[R, phi, Z] * b[R, phi, Z] \
             + Sqrt[2.0 * masses[[sigma]] * tempScalings[[sigma]]] \
               * vp / charges[[sigma]] * rhoref / Lref \
               * Curl[b[R, phi, Z], {R, phi, Z}, coordinatesystem]]

Bps[R_, phi_, Z_, vp_, sigma_] := If[equilibrium == "dommaschk",
    absBfunc[R, phi, Z], Simplify[b[R, phi, Z] . BstarES[R, phi, Z, vp, sigma]]]

dot[a_, b_] := a . b

(* Definition of the MMS functions *)

maxwellian[R_, phi_, Z_, vp_, mu_] := (Pi * T)^(-3 / 2) \
    * Exp[-1 / T * (vp^2 + mu * absBfunc[R, phi, Z])]

f[t_, R_, phi_, Z_, vp_, mu_] :=
    {0.95 * Sin[nrad * Pi * (rho[R, phi, Z] - rhomin) / (rhomax - rhomin)]^2 \
        * Sin[npol * theta[R, Z]]^2 * Cos[ntor * phi]^2 * Cos[omega * t]^2 \
        * maxwellian[R, phi, Z, vp, mu]
     + 0.05 * maxwellian[R, phi, Z, vp, mu], \
     0.95 * Sin[nrad * Pi * (rho[R, phi, Z] - rhomin) / (rhomax - rhomin)]^2 \
        * Sin[npol * theta[R, Z]]^2 * Cos[ntor * phi]^2 * Cos[omega * t]^2 \
        * maxwellian[R, phi, Z, vp, mu]
     + 0.05 * maxwellian[R, phi, Z, vp, mu]}

pot[t_, R_, phi_, Z_] :=
    Sin[2 * nrad * Pi * (rho[R, phi, Z] - rhomin) / (rhomax - rhomin)] \
        * Sin[2 * npol * theta[R, Z]] * Cos[ntor * phi]^2 * Cos[omega * t]^2

Apar[t_, R_, phi_, Z_] :=
    Sin[2 * nrad * Pi * (rho[R, phi, Z] - rhomin) / (rhomax - rhomin)] \
        * Sin[2 * npol * theta[R, Z]] * Cos[ntor * phi]^2 * Cos[omega * t]^2

Bpar[t_, R_, phi_, Z_] :=
    0.1 * Sin[2 * nrad * Pi * (rho[R, phi, Z] - rhomin) / (rhomax - rhomin)] \
        * Sin[2 * npol * theta[R, Z]] * Cos[ntor * phi]^2 * Cos[omega * t]^2

densi[t_, R_, phi_, Z_] :=
    0.95 * Sin[nrad * Pi * (rho[R, phi, Z] - rhomin) / (rhomax - rhomin)]^2 \
        * Sin[npol * theta[R, Z]]^2 * Cos[ntor * phi]^2 * Cos[omega * t]^2 \
    + 0.05

dense[t_, R_, phi_, Z_] :=
    0.95 * Sin[nrad * Pi * (rho[R, phi, Z] - rhomin) / (rhomax - rhomin)]^2 \
        * Sin[npol * theta[R, Z]]^2 * Cos[ntor * phi]^2 * Cos[omega * t]^2 \
    + 0.05

chargedens[t_, R_, phi_, Z_] :=
    Simplify[charges[[1]] * densi[t, R, phi, Z] \
             + charges[[2]] * dense[t, R, phi, Z]]

massdens[t_, R_, phi_, Z_] :=
    Simplify[masses[[1]] * densi[t, R, phi, Z] \
             + masses[[2]] * dense[t, R, phi, Z]]

currentdens[t_, R_, phi_, Z_] :=
    Simplify[charges[[1]] * betaref \
               * Sqrt[tempScalings[[1]] / (2 * masses[[1]])] * Pi \
               * Integrate[vp * f[t, R, phi, Z, vp, mu][[1]] \
                            * Bps[R, phi, Z, vp, 1], \
                          {vp, -Infinity, Infinity}, {mu, 0, Infinity}] \
             + charges[[2]] * betaref \
               * Sqrt[tempScalings[[2]] / (2 * masses[[2]])] * Pi \
               * Integrate[vp * f[t, R, phi, Z, vp, mu][[2]] \
                            * Bps[R, phi, Z, vp, 2], \
                          {vp, -Infinity, Infinity}, {mu, 0, Infinity}]]

perppressure[t_, R_, phi_, Z_] :=
    Simplify[-0.5 * betaref * (tempScalings[[1]] * Pi \
               * Integrate[mu * f[t, R, phi, Z, vp, mu][[1]] \
                            * Bps[R, phi, Z, vp, 1], \
                          {vp, -Infinity, Infinity}, {mu, 0, Infinity}] \
               + tempScalings[[2]] * Pi \
                 * Integrate[mu * f[t, R, phi, Z, vp, mu][[2]] \
                            * Bps[R, phi, Z, vp, 2], \
                          {vp, -Infinity, Infinity}, {mu, 0, Infinity}])]
(* Calculate derivatives *)

gradf[t_, R_, phi_, Z_, vp_, mu_, sigma_] :=
    Grad[f[t, R, phi, Z, vp, mu][[sigma]], {R, phi, Z}, coordinatesystem]

dfdvp[t_, R_, phi_, Z_, vp_, mu_, sigma_] :=
    D[f[t, R, phi, Z, vp, mu][[sigma]], vp]

dfdt[t_, R_, phi_, Z_, vp_, mu_, sigma_] :=
    D[f[t, R, phi, Z, vp, mu][[sigma]], t]

dApardt[t_, R_, phi_, Z_] := D[Apar[t, R, phi, Z], t]

gradApar[t_, R_, phi_, Z_] :=
    Grad[Apar[t, R, phi, Z], {R, phi, Z}, coordinatesystem]

gradpot[t_, R_, phi_, Z_] :=
    Grad[pot[t, R, phi, Z], {R, phi, Z}, coordinatesystem]

gradpot2[t_, R_, phi_, Z_] :=
    Grad[pot[t, R, phi, Z], {R, Z}, "Cartesian"] \
    . Grad[pot[t, R, phi, Z], {R, Z}, "Cartesian"];

gradH2[t_, R_, phi_, Z_, sigma_] := - (rhoref / Lref)^2 \
    * Grad[masses[[sigma]] / (2 * absBfunc[R, phi, Z]^2) \
            * gradpot2[t, R, phi, Z], {R, phi, Z}, coordinatesystem];

gradBpar[t_, R_, phi_, Z_] :=
    Grad[Bpar[t, R, phi, Z], {R, phi, Z}, coordinatesystem]

gradB[R_, phi_, Z_] := Grad[absBfunc[R, phi, Z], {R, phi, Z}, coordinatesystem]

laplacepot[t_, R_, phi_, Z_] :=
    -1 / jacobian[R] * Div[jacobian[R] * (rhoref / Lref)^2 \
        * massdens[t, R, phi, Z] / absBfunc[R, phi, Z]^2 \
        * Grad[pot[t, R, phi, Z], {R, Z}, "Cartesian"], {R, Z}, "Cartesian"]

laplaceApar[t_, R_, phi_, Z_] :=
    -1 / jacobian[R] * Div[jacobian[R] * (rhoref / Lref)^2 \
        * Grad[Apar[t, R, phi, Z], {R, Z}, "Cartesian"], {R, Z}, "Cartesian"]

(* Evaluate the Vlasov equation *)

BstarEM[t_, R_, phi_, Z_, vp_, sigma_] :=
     Simplify[BstarES[R, phi, Z, vp, sigma]
              + rhoref / Lref * Cross[gradApar[t, R, phi, Z], b[R, phi, Z]]]

bstaradv[t_, R_, phi_, Z_, vp_, mu_, sigma_] :=
    Sqrt[2 / masses[[sigma]]] * vp / Bps * \
    dot[BstarEM[t, R, phi, Z, vp, sigma], gradf[t, R, phi, Z, vp, mu, sigma]]

bcrossadv[t_, R_, phi_, Z_, vp_, mu_, sigma_] :=
     rhoref / (Lref * charges[[sigma]] * Bps) \
        * dot[Cross[b[R, phi, Z], mu * gradB[R, phi, Z] \
                + charges[[sigma]] * gradpot[t, R, phi, Z]], \
              gradf[t, R, phi, Z, vp, mu, sigma]]

(*H2 and B_par corrections*)
bcrossadv2[t_, R_, phi_, Z_, vp_, mu_, sigma_] :=
     rhoref / (Lref * charges[[sigma]] * Bps) \
        * dot[Cross[b[R, phi, Z], gradH2[t, R, phi, Z, sigma] \
                + tempScalings[[sigma]] * mu * gradBpar[t, R, phi, Z]], \
              gradf[t, R, phi, Z, vp, mu, sigma]]

vpadv[t_, R_, phi_, Z_, vp_, mu_, sigma_] :=
    ((-1.0 / (Sqrt[2.0 * masses[[sigma]]] * Bps) \
      * dot[BstarEM[t, R, phi, Z, vp, sigma], \
            mu * gradB[R, phi, Z] \
               + charges[[sigma]] * gradpot[t, R, phi, Z]] \
     ) - charges[[sigma]] / Sqrt[2.0 * masses[[sigma]]] \
           * dApardt[t, R, phi, Z] \
    ) * dfdvp[t, R, phi, Z, vp, mu, sigma]

(*H2 and B_par corrections*)
vpadv2[t_, R_, phi_, Z_, vp_, mu_, sigma_] :=
    (-1.0 / (Sqrt[2.0 * masses[[sigma]]] * Bps) \
      * dot[BstarEM[t, R, phi, Z, vp, sigma], \
            gradH2[t, R, phi, Z, sigma] \
               + tempScalings[[sigma]] * mu * gradBpar[t, R, phi, Z]] \
     ) * dfdvp[t, R, phi, Z, vp, mu, sigma]

(* Evaluate the Maxwell equations *)

sourcepot[t_, R_, phi_, Z_] :=
    laplacepot[t, R, phi, Z] - chargedens[t, R, phi, Z]

sourceApar[t_, R_, phi_, Z_] :=
    laplaceApar[t, R, phi, Z] - currentdens[t, R, phi, Z]

sourceBpar[t_, R_, phi_, Z_] :=
    Bpar[t, R, phi, Z] - perppressure[t, R, phi, Z]

sourceEpar[t_, R_, phi_, Z_] := D[sourceApar[t, R, phi, Z], t]

(* Write the result to file *)
Print[ElapsedTime <> "   writing to file in path ", path]

Print[ElapsedTime <> "      writing mms_solution_f"]

Export[StringJoin[path, "mms_solution_f_ions.txt"],
    {"mms_solution_f_ions = " <> F90Format[f[t, R, phi, Z, vp, mu][[1]]]}]

Export[StringJoin[path, "mms_solution_f_electrons.txt"],
    {"mms_solution_f_electrons = " <> F90Format[f[t, R, phi, Z, vp, mu][[2]]]}]

Print[ElapsedTime <> "      writing mms_solution_es_pot"]
Export[StringJoin[path, "mms_solution_es_pot.txt"],
    {"mms_solution_es_pot = " <> F90Format[pot[t, R, phi, Z]]}]

Print[ElapsedTime <> "      writing mms_solution_a_par"]
Export[StringJoin[path, "mms_solution_a_par.txt"],
    {"mms_solution_a_par = " <> F90Format[Apar[t, R, phi, Z]]}]

Print[ElapsedTime <> "      writing mms_solution_b_par"]
Export[StringJoin[path, "mms_solution_b_par.txt"],
    {"mms_solution_b_par = " <> F90Format[Bpar[t, R, phi, Z]]}]

Print[ElapsedTime <> "      writing mms_solution_e_par"]
Export[StringJoin[path, "mms_solution_e_par.txt"],
    {"mms_solution_e_par = " <> F90Format[dApardt[t, R, phi, Z]]}]

Print[ElapsedTime <> "      writing mms_magfield"]
(* Write absB and its derivatives to file, to be used
   implicitely in normb and the vlasov source files below. *)

Export[StringJoin[path, "mms_magfield.txt"],
    {"absB = " <> F90Format[absBfunc[R, phi, Z]] <>
     "dabsBdR = "   <> F90Format[D[absBfunc[R, phi, Z], R]] <>
     "dabsBdphi = " <> F90Format[D[absBfunc[R, phi, Z], phi]] <>
     "dabsBdZ = "   <> F90Format[D[absBfunc[R, phi, Z], Z]]}]

(* Clear absB and overwrite its Fortran form so it and its derivatives are
   written implicitely. For example, the derivative of absB with respect to R
   will be written dabsBdR. *)
Clear[absBfunc]
Unprotect[Derivative]
Derivative /: \
    Format[Derivative[dx1_, dx2_, dx3_][f_][x1_, x2_, x3_], FortranForm] := \
        ToExpression[StringJoin["d", ToString[f], \
                                "d", ToString[dx1 * x1 + dx2 * x2 + dx3 * x3]]]
Protect[Derivative]
(* To avoid unnecessary terms, overwrite absB with only the
   minimum dependencies on R, phi, and Z *)
absBfunc[R_, phi_, Z_] := Switch[equilibrium, "slab", 1, \
                                 "circular", absB[R, 0, Z], \
                                 "salpha", absB[R, 0, Z], \
                                 "dommaschk", absB[R, phi, Z]]
Unprotect[absB]
absB /: Format[absB[x1_, x2_, x3_], FortranForm] := absB
Protect[absB]

(* Write all components of b to file. Here only the partial derivates for
   Curl[b] are needed. *)
normb = b[R,phi,Z]
Export[OpenAppend[StringJoin[path, "mms_magfield.txt"]],
    {"bR = "   <> F90Format[normb[[1]]] <>
     "bphi = " <> F90Format[normb[[2]]] <>
     "bZ = "   <> F90Format[normb[[3]]] <>
     "dbRdphi = " <> F90Format[D[normb[[1]], phi]] <>
     "dbRdZ = "   <> F90Format[D[normb[[1]], Z]] <>
     "dbphidR = " <> F90Format[D[normb[[2]], R]] <>
     "dbphidZ = " <> F90Format[D[normb[[2]], Z]] <>
     "dbZdR = "   <> F90Format[D[normb[[3]], R]] <>
     "dbZdphi = " <> F90Format[D[normb[[3]], phi]]}]

(* To avoid unnecessary terms, overwrite b with only the
   minimum dependencies on R, phi, and Z *)
Clear[b]
b[R_, phi_, Z_] := Switch[equilibrium, \
    "slab",      {0,             1,               0}, \
    "circular",  {bR[R, 0, Z],   bphi[R, 0, Z],   bZ[R, 0, Z]}, \
    "salpha",    {bR[R, 0, Z],   bphi[R, 0, Z],   bZ[R, 0, Z]}, \
    "dommaschk", {bR[R, phi, Z], bphi[R, phi, Z], bZ[R, phi, Z]}]
Unprotect[bR]
bR /: Format[bR[x1_, x2_, x3_], FortranForm] := bR
Protect[bR]
Unprotect[bphi]
bphi /: Format[bphi[x1_, x2_, x3_], FortranForm] := bphi
Protect[bphi]
Unprotect[bZ]
bZ /: Format[bZ[x1_, x2_, x3_], FortranForm] := bZ
Protect[bZ]

Print[ElapsedTime <> "      writing mms_density"]
Export[StringJoin[path, "mms_density_ions.txt"],
    {"mms_density_ions = " <> F90Format[densi[t, R, phi, Z]]}]
Export[StringJoin[path, "mms_density_electrons.txt"],
    {"mms_density_electrons = " <> F90Format[dense[t, R, phi, Z]]}]

Print[ElapsedTime <> "      writing mms_source_es_pot"]
Export[StringJoin[path, "mms_source_es_pot.txt"],
    {"mms_source_es_pot = " <> F90Format[sourcepot[t, R, phi, Z]]}]

Print[ElapsedTime <> "      writing mms_source_a_par"]
Export[StringJoin[path, "mms_source_a_par.txt"],
    {"mms_source_a_par = " <> F90Format[sourceApar[t, R, phi, Z]]}]

Print[ElapsedTime <> "      writing mms_source_b_par"]
Export[StringJoin[path, "mms_source_b_par.txt"],
    {"mms_source_b_par = " <> F90Format[sourceBpar[t, R, phi, Z]]}]

Print[ElapsedTime <> "      writing mms_source_e_par"]
Export[StringJoin[path, "mms_source_e_par.txt"],
    {"mms_source_e_par = " <> F90Format[sourceEpar[t, R, phi, Z]]}]

(* Write MMS vlasov sources *)

Print[ElapsedTime <> "      writing mms_source_ions"]
Export[StringJoin[path, "mms_source_ions.txt"],
     {"Bps = " <> F90Format[Bps[R, phi, Z, vp, 1]] <>
      "dfdt = " <> F90Format[dfdt[t, R, phi, Z, vp, mu, 1]] <>
      "vpadv = " <> F90Format[vpadv[t, R, phi, Z, vp, mu, 1]] <>
      "vpadv2 = " <> F90Format[vpadv2[t, R, phi, Z, vp, mu, 1]] <>
      "bstaradv = " <> F90Format[bstaradv[t, R, phi, Z, vp, mu, 1]] <>
      "bcrossadv = " <> F90Format[bcrossadv[t, R, phi, Z, vp, mu, 1]] <>
      "bcrossadv2 = " <> F90Format[bcrossadv2[t, R, phi, Z, vp, mu, 1]] <>
      "mms_source_ions = dfdt + vpadv + vpadv2 + bstaradv + bcrossadv " <>
                         "+ bcrossadv2\n"}]

Print[ElapsedTime <> "      writing mms_source_electrons"]
Export[StringJoin[path, "mms_source_electrons.txt"],
     {"Bps = " <> F90Format[Bps[R, phi, Z, vp, 2]] <>
      "dfdt = " <> F90Format[dfdt[t, R, phi, Z, vp, mu, 2]] <>
      "vpadv = " <> F90Format[vpadv[t, R, phi, Z, vp, mu, 2]] <>
      "vpadv2 = " <> F90Format[vpadv2[t, R, phi, Z, vp, mu, 2]] <>
      "bstaradv = " <> F90Format[bstaradv[t, R, phi, Z, vp, mu, 2]] <>
      "bcrossadv = " <> F90Format[bcrossadv[t, R, phi, Z, vp, mu, 2]] <>
      "bcrossadv2 = " <> F90Format[bcrossadv2[t, R, phi, Z, vp, mu, 2]] <>
      "mms_source_electrons = dfdt + vpadv + vpadv2 + bstaradv " <>
                              "+ bcrossadv + bcrossadv2\n"}]

Print[ElapsedTime <> "   file write complete"]
TotalTime = TextString[TimeObject[{0, 0, TimeUsed[] - t1}],
                       TimeFormat->{"Hour","h ","Minute","m ","Second","s"}]
Print["Total execution time: ", TotalTime]
