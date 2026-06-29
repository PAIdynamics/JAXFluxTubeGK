BeginPackage["DommaschkEquilibrium`"]

(* To run the package, execute the following line in a notebook
<< DommaschkEquilibrium`
*)

(* Provides magnetic field for the Dommaschk equilibrium according to [1].

   Either the specific equilibrium in Eq. 39 (Fig2a) of [1], or the equilibrium
   called dom25b in [2] (Table IV and Fig. 41) can be realized here. The user
   must uncomment the relevant sections at the bottom of the script.

   [1] W. Dommaschk, Representations for vacuum potentials in stellarators,
   Computer Physics Communications 40, pg. 203 (1986)
   [2] W. Dommaschk et al., Detailed results of Monte Carlo simulation of
   neoclassical transport in stellarators, IPP internal report 0/48, (1984) *)

(* Basic Factors *)
AlphaN[m_, n_] := 0 /; n < 0
AlphaN[m_, n_] := (-1)^n / (Gamma[m + n + 1] * Gamma[n + 1] * 2^(2 * n + m))

AlphaSt[m_, n_] := 0 /; n < 0
AlphaSt[m_, n_] := (2 * n + m) * AlphaN[m, n]

BetaN[m_, n_] := 0 /; n >= m
BetaN[m_, n_] := 0 /; n < 0
BetaN[m_, n_] := Gamma[m - n] / (Gamma[n + 1] * 2^(2 * n - m + 1))

BetaSt[m_, n_] := 0 /; n >= m
BetaSt[m_, n_] := 0 /; n < 0
BetaSt[m_, n_] := (2 * n - m) * BetaN[m, n]

GammaN[m_, n_] := 0 /; n < 0
GammaN[m_, n_] := AlphaN[m, n] / 2 \
                  * (Sum[1 / i, {i, 1, n}] + Sum[1 / i, {i, 1, n + m}])

GammaSt[m_, n_] := 0 /; n < 0
GammaSt[m_, n_] := (2 * n + m) * GammaN[m, n]

(* Equations 31 and 32 [1] *)
CDMK[R_, m_, k_] := Sum[-(AlphaN[m, j] * (AlphaSt[m, k - m - j] * Log[R] \
                          + GammaSt[m, k - m - j] - AlphaN[m, k - m - j]) \
                          - GammaN[m, j] * AlphaSt[m, k - m - j] \
                          + AlphaN[m, j] * BetaSt[m, k - j]) * R^(2 * j + m) \
                        + BetaN[m, j] * AlphaSt[m, k - j] * R^(2 * j - m), \
                        {j, 0, k}]
CNMK[R_, m_, k_] := Sum[(AlphaN[m, j] * (AlphaN[m, k - m - j] * Log[R] \
                                         + GammaN[m, k - m - j]) \
                         - GammaN[m, j] * AlphaN[m, k - m - j] \
                         + AlphaN[m, j] * BetaN[m, k - j]) * R^(2 * j + m) \
                         - BetaN[m, j] * AlphaN[m, k - j] * R^(2 * j - m), \
                         {j, 0, k}]

(* Equations 8 and 9 [1] *)
DMN[R_, Z_, m_, n_] := Sum[Z^(n - 2 * k) \
                           / Factorial[n - 2 * k] * CDMK[R, m, k], \
                           {k, 0, Floor[n / 2]}]
NMN[R_, Z_, m_, n_] := Sum[Z^(n - 2 * k) \
                           / Factorial[n - 2 * k] * CNMK[R, m, k], \
                           {k, 0, Floor[n / 2]}]

(* Equation 12 [1] *)
VML[R_, phi_, Z_, m_, l_] := ( cfa[[m, l]] * Cos[m * phi] \
                             + cfb[[m, l]] * Sin[m * phi]) * DMN[R, Z, m, l] \
                           + ( cfc[[m, l]] * Cos[m * phi] \
                             + cfd[[m, l]] * Sin[m * phi]) * NMN[R, Z, m, l - 1]

(* Uncomment this section for the equilibrium of Eq. 39 (Fig2a) of [1] *)
mtor = 5;
lpol = 4;
cfa = ConstantArray[0, {mtor, lpol}];
cfb = ConstantArray[0, {mtor, lpol}];
cfc = ConstantArray[0, {mtor, lpol}];
cfd = ConstantArray[0, {mtor, lpol}];
cfb[[5, 2]] = 14 / 10;
cfc[[5, 2]] = 14 / 10;
cfb[[5, 4]] = 1925 / 100;
Vtot[R_, phi_, Z_] = VML[R, phi, Z, 5, 2] + VML[R, phi, Z, 5, 4];

(* Uncomment this section for the equilibrium dom25b in [2] *)
(*
mtor = 5;
lpol = 2;
cfa = ConstantArray[0, {mtor, lpol}];
cfb = ConstantArray[0, {mtor, lpol}];
cfc = ConstantArray[0, {mtor, lpol}];
cfd = ConstantArray[0, {mtor, lpol}];
cfb[[5, 2]] = 1489 / 1000;
cfc[[5, 2]] = 1489 / 1000;
Vtot[R_, phi_, Z_] = VML[R, phi, Z, 5, 2];
*)

(* Magnetic field components *)
BDommaschk[R_, phi_, Z_] = {D[Vtot[R, phi, Z], R],
                            1 / R + 1 / R * D[Vtot[R, phi, Z], phi],
                            D[Vtot[R, phi, Z], Z]};
absBDommaschk[R_, phi_, Z_] = \
                            Sqrt[BDommaschk[R, phi, Z] . BDommaschk[R, phi, Z]];

EndPackage[]
