BeginPackage["Fortran90`"]

(* This package gives the ability to take a Mathematica expression and convert
   it into a string compatable with Fortan 90 code. This is done using the
   intrinsic Mathematica routine FortranForm with minor modifications.
   The expression is returned split into lines with the line break character &,
   with lines no longer than 80 characters. *)

(* Write real numbers in scientific notation with the powers of 10 indicated by
   D. E.g. 156.42 becomes 1.5642D2 *)
Unprotect[D]
Clear[D]
Protect[D]
Unprotect[Real]
RTogle=False
Real /: Format[r_Real /; r >= 0, FortranForm] /; (RTogle =! RTogle) :=
    Module[{mantissa, exponent, tmp},
      {mantissa, exponent} = MantissaExponent[r];
      If[r === 0., exponent = 1];
      SequenceForm[10 mantissa, D, exponent - 1]
    ];
Protect[Real]

(* Overwrite the FortranForm of certain functions to be compatible with F90 *)
Unprotect[Power]
Power /: Format[Power[E, x_], FortranForm] := exp[x]
Protect[Power]
Unprotect[Csc]
Csc /: Format[Csc[x_], FortranForm] := 1.0 / sin[x]
Protect[Csc]
Unprotect[ArcTan]
ArcTan /: Format[ArcTan[x_, y_], FortranForm] := atan[y, x]
Protect[ArcTan]

(* Return a string with the F90 form of a given expression. Lines in the
   string are broken before the 80 character limit. *)
F90Format[expression_] :=
    Module[{tmp, splits},
      tmp = FortranForm[expression];
      tmp = ToString[tmp, PageWidth -> Infinity, TotalWidth -> Infinity];
      tmp = StringReplace[tmp, "Sqrt(2)" -> "Sqrt(2.0_GP)"];
      tmp = StringReplace[tmp, "Sqrt(3)" -> "Sqrt(3.0_GP)"];
      tmp = StringReplace[tmp, "Sqrt(4)" -> "Sqrt(4.0_GP)"];
      tmp = StringReplace[tmp, "Sqrt(5)" -> "Sqrt(5.0_GP)"];
      tmp = StringReplace[tmp, "Sqrt(6)" -> "Sqrt(6.0_GP)"];
      tmp = StringReplace[tmp, "Cos(1)" -> "Cos(1.0_GP)"];
      tmp = StringReplace[tmp, "Cos(2)" -> "Cos(2.0_GP)"];
      tmp = StringReplace[tmp, "Cos(3)" -> "Cos(3.0_GP)"];
      tmp = StringReplace[tmp, "Cos(4)" -> "Cos(4.0_GP)"];
      tmp = StringReplace[tmp, "Cos(5)" -> "Cos(5.0_GP)"];
      tmp = StringReplace[tmp, "Cos(6)" -> "Cos(6.0_GP)"];
      tmp = StringReplace[tmp, "KroneckerDelta" -> "krond"];
      splits = StringSplit[tmp, "(" -> "("];
      tmp = Fold[If[StringLength[Last @ #1] + StringLength[#2] > 80, \
                    Join[#1, {#2}], \
                    Join[Most[#1], {StringJoin[Last[#1], #2]}]
                   ]&,
                 {First @ splits}, Rest[splits]];
      tmp = StringJoin @@ Riffle[tmp, " &\n"];
      tmp = StringJoin[tmp, "\n"];
      Return[tmp];
    ];

EndPackage[]
