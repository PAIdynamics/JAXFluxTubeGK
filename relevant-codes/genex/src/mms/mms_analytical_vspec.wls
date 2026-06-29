#!/usr/bin/env wolframscript
(* This script calculates solutions and sources for the MMS tests for the
    spectral approach.
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
T := 1
Tmaxw := 1

(* Define sum upper bounds *)
ubvp := 3
ubmu := 1

(* Definition of the coordinate system and magnetic field *)
coordinatesystem = If[equilibrium == "salpha" ||
                      equilibrium == "dommaschk", "Cylindrical", "Cartesian"]

(* Parameters for Dommaschk rotated ellipse approximation for rho *)
axis1 := 4.296 * 10^-2
axis2 := 2.161 * 10^-2
h := 0.999
k := 0.0
p := 2.5
rho[R_, phi_, Z_] := Switch[equilibrium, "slab", R, \
    "circular", CoordinateTransform["Cartesian" -> "Polar", {R, Z}][[1]], \
    "salpha", CoordinateTransform["Cartesian" -> "Polar", \
                                  {R - 1.0, Z}][[1]] / minorr, \
    "dommaschk", Sqrt[ ( (R - h) * Cos[p * phi] \
                       + (Z - k) * Sin[p * phi])^2 / axis1^2 \
                     + ( (R - h) * Sin[p * phi] \
                       - (Z - k) * Cos[p * phi])^2 / axis2^2]]

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

dot[a_, b_] := a . b

(* Definition of the MMS vspec functions *)

pot[t_, R_, phi_, Z_] :=
    Sin[2 * nrad * Pi * (rho[R, phi, Z] - rhomin) / (rhomax - rhomin)] \
        * Sin[2 * npol * theta[R, Z]] * Cos[ntor * phi]^2 * Cos[omega * t]^2

Apar[t_, R_, phi_, Z_] :=
    Sin[2 * nrad * Pi * (rho[R, phi, Z] - rhomin) / (rhomax - rhomin)] \
        * Sin[2 * npol * theta[R, Z]] * Cos[ntor * phi]^2 * Cos[omega * t]^2

fRphiZ[t_, R_, phi_, Z_] := \
        0.95 \
        * Sin[nrad * Pi * (rho[R, phi, Z] - rhomin) / (rhomax - rhomin)]^2 \
        * Sin[npol * theta[R, Z]]^2 * Cos[ntor * phi]^2 * Cos[omega * t]^2 \
        + 0.05

vspec[vp_, mu_] := Cos[vp + 2 * mu] \
                   * (KroneckerDelta[int[vp], 0] * KroneckerDelta[int[mu], 0] \
                   + KroneckerDelta[int[vp], 1] * KroneckerDelta[int[mu], 0] \
                   + KroneckerDelta[int[vp], 2] * KroneckerDelta[int[mu], 0] \
                   + KroneckerDelta[int[vp], 3] * KroneckerDelta[int[mu], 0] \
                   + KroneckerDelta[int[vp], 0] * KroneckerDelta[int[mu], 1] \
                   + KroneckerDelta[int[vp], 1] * KroneckerDelta[int[mu], 1] \
                   + KroneckerDelta[int[vp], 2] * KroneckerDelta[int[mu], 1] \
                   + KroneckerDelta[int[vp], 3] * KroneckerDelta[int[mu], 1])

fvspec[t_, R_, phi_, Z_, vp_, mu_] := {fRphiZ[t, R, phi, Z] * vspec[vp, mu], \
                                       fRphiZ[t, R, phi, Z] * vspec[vp, mu]}

densi[t_, R_, phi_, Z_] := fRphiZ[t, R, phi, Z]

dense[t_, R_, phi_, Z_] := fRphiZ[t, R, phi, Z]

chargedens[t_, R_, phi_, Z_] :=
    Simplify[charges[[1]] * densi[t, R, phi, Z] \
             + charges[[2]] * dense[t, R, phi, Z]]

massdens[t_, R_, phi_, Z_] :=
    Simplify[masses[[1]] * densi[t, R, phi, Z] \
             + masses[[2]] * dense[t, R, phi, Z]]

upari[t_, R_, phi_, Z_] := fvspec[t, R, phi, Z, 1, 0][[1]]

upare[t_, R_, phi_, Z_] := fvspec[t, R, phi, Z, 1, 0][[2]]

currentdens[t_, R_, phi_, Z_] := betaref / 2 \
          * (
          charges[[1]] * Sqrt[Tmaxw / masses[[1]]] \
          * upari[t, R, phi, Z] \
          + charges[[2]] * Sqrt[Tmaxw / masses[[2]]] \
          * upare[t, R, phi, Z])

(* Calculate derivatives *)

gradfRphiZ[t_, R_, phi_, Z_] := Grad[fRphiZ[t, R, phi, Z], \
                                     {R, phi, Z}, \
                                     coordinatesystem]

dfdt[t_, R_, phi_, Z_, vp_, mu_, sigma_] :=
    D[fvspec[t, R, phi, Z, vp, mu][[sigma]], t]

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

gradpottot[t_, R_, phi_, Z_, sigma_] := gradpot[t, R, phi, Z] \
                              + gradH2[t, R, phi, Z, sigma] / charges[[sigma]]

gradB[R_, phi_, Z_] := Grad[absBfunc[R, phi, Z], {R, phi, Z}, coordinatesystem]

curlb[R_, phi_, Z_] := Curl[b[R, phi, Z], {R, phi, Z}, coordinatesystem]

laplacepot[t_, R_, phi_, Z_] :=
    -1 / jacobian[R] * Div[jacobian[R] * (rhoref / Lref)^2 \
        * massdens[t, R, phi, Z] / absBfunc[R, phi, Z]^2 \
        * Grad[pot[t, R, phi, Z], {R, Z}, "Cartesian"], {R, Z}, "Cartesian"]

laplaceApar[t_, R_, phi_, Z_] :=
    -1 / jacobian[R] * Div[jacobian[R] * (rhoref / Lref)^2 \
        * Grad[Apar[t, R, phi, Z], {R, Z}, "Cartesian"], {R, Z}, "Cartesian"]

(* Evaluate the spectral Vlasov equation *)

dynamicvspec[t_, R_, phi_, Z_, vp_, mu_, sigma_] :=  \
            dApardt[t, R, phi, Z] \
            * charges[[sigma]] / Sqrt[Tmaxw * masses[[sigma]]] \
            * sqrt[vp] * fvspec[t, R, phi, Z, vp - 1, mu][[sigma]]

a1vspec[vp_, mu_] := Sum[(mu * Sqrt[vp] \
                     * KroneckerDelta[int[vp - 1], ell] \
                     * KroneckerDelta[int[mu], k] \
                    - mu * Sqrt[vp] \
                     * KroneckerDelta[int[vp - 1], ell] \
                     * KroneckerDelta[int[mu + 1], k] \
                    - mu * Sqrt[vp + 1] \
                     * KroneckerDelta[int[vp + 1], ell] \
                     * KroneckerDelta[int[mu], k] \
                    + mu * Sqrt[vp + 1] \
                     * KroneckerDelta[int[vp + 1], ell] \
                     * KroneckerDelta[int[mu - 1], k] \
                    - Sqrt[vp] \
                     * KroneckerDelta[int[vp - 1], ell] \
                     * KroneckerDelta[int[mu + 1], k] \
                    - Sqrt[vp + 1] \
                     * KroneckerDelta[int[vp + 1], ell] \
                     * KroneckerDelta[int[mu], k]) \
                     * vspec[ell, k], \
                     {ell, 0, ubvp}, {k, 0, ubmu}]

a1[t_, R_, phi_, Z_, vp_, mu_, sigma_] := fRphiZ[t, R, phi, Z] * ( \
         Sqrt[Tmaxw / masses[[sigma]]] / absBfunc[R, phi, Z] \
          * (dot[b[R, phi, Z], gradB[R, phi, Z]] \
         + rhoref / Lref / absBfunc[R, phi, Z] \
          * dot[Cross[b[R, phi, Z], gradB[R, phi, Z]], \
               gradApar[t, R, phi, Z]]) \
          * a1vspec[vp, mu])

a2vspec[vp_, mu_] := Sum[(mu * Sqrt[vp * (vp - 1)] \
                     * KroneckerDelta[int[vp - 2], ell] \
                     * KroneckerDelta[int[mu], k] \
                    - (mu + 1) * Sqrt[vp * (vp - 1)] \
                     * KroneckerDelta[int[vp - 2], ell] \
                     * KroneckerDelta[int[mu + 1], k] \
                    + mu * vp \
                     * KroneckerDelta[int[vp], ell] \
                     * KroneckerDelta[int[mu - 1], k] \
                    - (mu + 1) * Sqrt[(vp + 1) * (vp + 2)] \
                     * KroneckerDelta[int[vp + 2], ell] \
                     * KroneckerDelta[int[mu], k] \
                    + mu * Sqrt[(vp + 1) * (vp + 2)]
                     * KroneckerDelta[int[vp + 2], ell] \
                     * KroneckerDelta[int[mu - 1], k] \
                    + (mu - vp) \
                     * KroneckerDelta[int[vp], ell] \
                     * KroneckerDelta[int[mu], k] \
                    - (mu + 1) * (vp + 1) \
                     * KroneckerDelta[int[vp], ell] \
                     * KroneckerDelta[int[mu + 1], k]) \
                     * vspec[ell, k], \
                     {ell, 0, ubvp}, {k, 0, ubmu}]

a2[t_, R_, phi_, Z_, vp_, mu_, sigma_] := fRphiZ[t, R, phi, Z] * ( \
          Tmaxw / absBfunc[R, phi, Z]^2 / charges[[sigma]] \
          * rhoref / Lref * dot[curlb[R, phi, Z], gradB[R, phi, Z]] \
          * a2vspec[vp, mu])

a3vspec[vp_, mu_] := Sum[(mu * KroneckerDelta[int[vp], ell] \
                     * KroneckerDelta[int[mu - 1], k] \
                     - (mu + 1) * KroneckerDelta[int[vp], ell] \
                     * KroneckerDelta[int[mu], k]) \
                     * vspec[ell, k], \
                     {ell, 0, ubvp}, {k, 0, ubmu}]

a3[t_, R_, phi_, Z_, vp_, mu_, sigma_] := fRphiZ[t, R, phi, Z] * ( \
          1 / absBfunc[R, phi, Z]^2 * rhoref / Lref \
          * dot[Cross[b[R, phi, Z], gradpottot[t, R, phi, Z, sigma]], \
                gradB[R, phi, Z]] \
          * a3vspec[vp, mu])

a4vspec[vp_, mu_] := Sum[((vp + 1) * KroneckerDelta[ell, int[vp]] \
                      * KroneckerDelta[k, int[mu]] \
                      + sqrthv[vp * (vp - 1)] \
                      * KroneckerDelta[int[mu], k] \
                      * KroneckerDelta[ell, int[vp - 2]]) \
                      * vspec[ell, k], \
                      {ell, 0, ubvp}, {k, 0, ubmu}]

a4[t_, R_, phi_, Z_, vp_, mu_, sigma_] := fRphiZ[t, R, phi, Z] * ( \
          rhoref / Lref / absBfunc[R, phi, Z] \
          * dot[curlb[R, phi, Z], gradpottot[t, R, phi, Z, sigma]] \
          * a4vspec[vp, mu])

a5vspec[vp_, mu_] := Sum[(Sqrt[vp] \
                      * KroneckerDelta[int[mu], k] \
                      * KroneckerDelta[ell, int[vp - 1]])
                      * vspec[ell, k], \
                      {ell, 0, ubvp}, {k, 0, ubmu}]

a5[t_, R_, phi_, Z_, vp_, mu_, sigma_] := fRphiZ[t, R, phi, Z] * ( \
          charges[[sigma]] / Sqrt[Tmaxw * masses[[sigma]]] \
          * (dot[b[R, phi, Z], gradpottot[t, R, phi, Z, sigma]] \
             + rhoref / Lref / absBfunc[R, phi, Z] \
             * dot[Cross[b[R, phi, Z], gradpottot[t, R, phi, Z, sigma]], \
                   gradApar[t, R, phi, Z]]) \
          * a5vspec[vp, mu])

I00[vp_, mu_] := Sum[KroneckerDelta[int[vp], ell] \
                              * KroneckerDelta[int[mu], k] \
                              * vspec[ell, k], \
                              {ell, 0, ubvp}, {k, 0, ubmu}]

I10[vp_, mu_] := Sum[(sqrthv[(vp + 1) / 2] \
                     * KroneckerDelta[int[vp + 1], ell] \
                     + sqrthv[vp / 2] \
                     * KroneckerDelta[int[vp - 1], ell]) \
                     * KroneckerDelta[int[mu], k] \
                     * vspec[ell, k], \
                     {ell, 0, ubvp}, {k, 0, ubmu}]

I20[vp_, mu_] := sqrthv[(vp + 1) / 2] * I10[vp + 1, mu] \
               + sqrthv[vp / 2] * I10[vp - 1, mu]

I01[vp_, mu_] := Sum[KroneckerDelta[int[vp], ell] \
                     *((2 * mu + 1) * KroneckerDelta[int[mu], k] \
                     - mu * KroneckerDelta[int[mu - 1], k] \
                     - (mu + 1) * KroneckerDelta[int[mu + 1], k]) \
                     * vspec[ell, k], \
                     {ell, 0, ubvp}, {k, 0, ubmu}]

I11[vp_, mu_] := sqrthv[(vp + 1) / 2] * I01[vp + 1, mu] \
               + sqrthv[vp/2] * I01[vp - 1, mu]

gradI10[t_, R_, phi_, Z_, vp_, mu_] := I10[vp, mu] \
                     * (dot[b[R, phi, Z], gradfRphiZ[t, R, phi, Z]] \
                     + rhoref / Lref / absBfunc[R, phi, Z] \
                     * dot[Cross[b[R, phi, Z], gradfRphiZ[t, R, phi, Z]], \
                           gradApar[t, R, phi, Z]])

curvI20[t_, R_, phi_, Z_, vp_, mu_] := I20[vp, mu] \
                        * dot[curlb[R, phi, Z], gradfRphiZ[t, R, phi, Z]]

pbabsBI01[t_, R_, phi_, Z_, vp_, mu_] := I01[vp, mu] \
                    * dot[Cross[b[R, phi, Z], gradB[R, phi, Z]], \
                          gradfRphiZ[t, R, phi, Z]]

pbphiI00[t_, R_, phi_, Z_, vp_, mu_, sigma_] := I00[vp, mu] \
                  * dot[Cross[b[R, phi, Z], gradpottot[t, R, phi, Z, sigma]], \
                        gradfRphiZ[t, R, phi, Z]]

b1[t_, R_, phi_, Z_, vp_, mu_, sigma_] := \
                    Sqrt[2 * Tmaxw / masses[[sigma]]] \
                    * gradI10[t, R, phi, Z, vp, mu]

b2[t_, R_, phi_, Z_, vp_, mu_, sigma_] := \
                    2 * Tmaxw / absBfunc[R, phi, Z] / charges[[sigma]] \
                    * rhoref / Lref * curvI20[t, R, phi, Z, vp, mu]

b3[t_, R_, phi_, Z_, vp_, mu_, sigma_] := \
                    Tmaxw / charges[[sigma]] / absBfunc[R, phi, Z]^2 \
                    * rhoref / Lref * pbabsBI01[t, R, phi, Z, vp, mu]

b4[t_, R_, phi_, Z_, vp_, mu_, sigma_] := \
                    rhoref / Lref / absBfunc[R, phi, Z] \
                    * pbphiI00[t, R, phi, Z, vp, mu, sigma]

(* Evaluate the spectral Maxwell equations *)

sourcepot[t_, R_, phi_, Z_] :=
    laplacepot[t, R, phi, Z] - chargedens[t, R, phi, Z]

sourceApar[t_, R_, phi_, Z_] :=
    laplaceApar[t, R, phi, Z] - currentdens[t, R, phi, Z]

sourceEpar[t_, R_, phi_, Z_] := D[laplaceApar[t, R, phi, Z] - betaref / 2 \
          * (
          charges[[1]] * Sqrt[Tmaxw / masses[[1]]] \
          * upari[t, R, phi, Z] \
          + charges[[2]] * Sqrt[Tmaxw / masses[[2]]] \
          * upare[t, R, phi, Z]), t]

(* Write the result to file *)
Print[ElapsedTime <> "   writing to file in path ", path]

Print[ElapsedTime <> "      writing mms_solution_f"]

Export[StringJoin[path, "mms_solution_f_vspec_ions.txt"],
    {"mms_solution_f_ions = " <>
     F90Format[fvspec[t, R, phi, Z, vp, mu][[1]]]}]

Export[StringJoin[path, "mms_solution_f_vspec_electrons.txt"],
  {"mms_solution_f_electrons = " <>
   F90Format[fvspec[t, R, phi, Z, vp, mu][[2]]]}]

Print[ElapsedTime <> "      writing mms_solution_es_pot"]
Export[StringJoin[path, "mms_solution_es_pot.txt"],
    {"mms_solution_es_pot = " <> F90Format[pot[t, R, phi, Z]]}]

Print[ElapsedTime <> "      writing mms_solution_a_par"]
Export[StringJoin[path, "mms_solution_a_par.txt"],
    {"mms_solution_a_par = " <> F90Format[Apar[t, R, phi, Z]]}]

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
Export[StringJoin[path, "mms_density_vspec_ions.txt"],
    {"mms_density_ions = " <> F90Format[densi[t, R, phi, Z]]}]
Export[StringJoin[path, "mms_density_vspec_electrons.txt"],
    {"mms_density_electrons = " <> F90Format[dense[t, R, phi, Z]]}]

Print[ElapsedTime <> "      writing mms_source_es_pot"]
Export[StringJoin[path, "mms_source_vspec_es_pot.txt"],
    {"mms_source_es_pot = " <> F90Format[sourcepot[t, R, phi, Z]]}]

Print[ElapsedTime <> "      writing mms_source_a_par"]
Export[StringJoin[path, "mms_source_vspec_a_par.txt"],
    {"mms_source_a_par = " <> F90Format[sourceApar[t, R, phi, Z]]}]

Print[ElapsedTime <> "      writing mms_source_e_par"]
Export[StringJoin[path, "mms_source_vspec_e_par.txt"],
    {"mms_source_e_par = " <> F90Format[sourceEpar[t, R, phi, Z]]}]

(* Write MMS spectral vlasov sources *)

Print[ElapsedTime <> "      writing mms_source_ions"]
Export[StringJoin[path, "mms_source_vspec_ions.txt"],
     {"dfdt = " <> F90Format[dfdt[t, R, phi, Z, vp, mu, 1]] <>
      "dynamicvspec = " <> F90Format[dynamicvspec[t, R, phi, Z, vp, mu, 1]] <>
      "a1 = " <> F90Format[a1[t, R, phi, Z, vp, mu, 1]] <>
      "a2 = " <> F90Format[a2[t, R, phi, Z, vp, mu, 1]] <>
      "a3 = " <> F90Format[a3[t, R, phi, Z, vp, mu, 1]] <>
      "a4 = " <> F90Format[a4[t, R, phi, Z, vp, mu, 1]] <>
      "a5 = " <> F90Format[a5[t, R, phi, Z, vp, mu, 1]] <>
      "b1 = " <> F90Format[b1[t, R, phi, Z, vp, mu, 1]] <>
      "b2 = " <> F90Format[b2[t, R, phi, Z, vp, mu, 1]] <>
      "b3 = " <> F90Format[b3[t, R, phi, Z, vp, mu, 1]] <>
      "b4 = " <> F90Format[b4[t, R, phi, Z, vp, mu, 1]] <>
      "alkpj = a1 + a2 + a3 + a4 + a5 \n" <>
      "blkpj = b1 + b2 + b3 + b4 \n" <>
      "mms_source_ions = dfdt + alkpj + blkpj + dynamicvspec"}]

Print[ElapsedTime <> "      writing mms_source_electrons"]
Export[StringJoin[path, "mms_source_vspec_electrons.txt"],
     {"dfdt = " <> F90Format[dfdt[t, R, phi, Z, vp, mu, 2]] <>
      "dynamicvspec = " <> F90Format[dynamicvspec[t, R, phi, Z, vp, mu, 2]] <>
      "a1 = " <> F90Format[a1[t, R, phi, Z, vp, mu, 2]] <>
      "a2 = " <> F90Format[a2[t, R, phi, Z, vp, mu, 2]] <>
      "a3 = " <> F90Format[a3[t, R, phi, Z, vp, mu, 2]] <>
      "a4 = " <> F90Format[a4[t, R, phi, Z, vp, mu, 2]] <>
      "a5 = " <> F90Format[a5[t, R, phi, Z, vp, mu, 2]] <>
      "b1 = " <> F90Format[b1[t, R, phi, Z, vp, mu, 2]] <>
      "b2 = " <> F90Format[b2[t, R, phi, Z, vp, mu, 2]] <>
      "b3 = " <> F90Format[b3[t, R, phi, Z, vp, mu, 2]] <>
      "b4 = " <> F90Format[b4[t, R, phi, Z, vp, mu, 2]] <>
      "alkpj = a1 + a2 + a3 + a4 + a5 \n" <>
      "blkpj = b1 + b2 + b3 + b4 \n" <>
      "mms_source_electrons = dfdt + alkpj + blkpj + dynamicvspec"}]

Print[ElapsedTime <> "   file write complete"]
TotalTime = TextString[TimeObject[{0, 0, TimeUsed[] - t1}],
                       TimeFormat->{"Hour","h ","Minute","m ","Second","s"}]
Print["Total execution time: ", TotalTime]