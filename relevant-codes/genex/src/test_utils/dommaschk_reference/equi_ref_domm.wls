#!/usr/bin/env wolframscript

(* Include Packages *)

CWD = Directory[]
SetDirectory["../mathematica_packages"]
(* Import the F90Format function *)
<< Fortran90`
(* Import the absBDommaschk and BDommaschk functions *)
<< DommaschkEquilibrium`
SetDirectory[CWD]

Print["Beginning Dommaschk field calculations"]
absB = absBDommaschk[x, phi, z];
(* Magnetic field unit vector {b_R, b_phi, b_Z} *)
b = BDommaschk[x, phi, z] / absB;
(* For Dommaschk, curlnormby is analytically zero. This is recovered by
   Mathematica with the Simplify. *)
curlnormby = Simplify[b . Curl[b, {x, phi, z}, "Cylindrical"]];
dabsBdx = D[absB, x];
dabsBdy = b . Grad[absB, {x, phi, z}, "Cylindrical"];
dabsBdz = D[absB, z];
(* For the mesh unit test, these quantities are only
   checked on a plane, i.e. when the local cartesian
   coordinate yc = 0. In this case the coordinates
   (xc, zc) coincide with the cylindrical coordinates (R, Z).
   Therefore g_yx = b_xc = b_R and g_yz = b_zc = b_Z. *)
dgyzdx = D[b[[3]], x];
dgyxdz = D[b[[1]], z];
dgyzdz = D[b[[3]], z];
dgyxdx = D[b[[1]], x];
(* Directional derivative of b_xc along b. Since
   b_xc = b_R cos(phi') - b_phi sin(phi'), where
   phi' = arctan(yc / xc) is the angle from the current
   mesh, the phi derivative of b_xc (evaluated at phi' = 0)
   not only includes the phi derivative of b_R, but also a
   term -b_phi. *)
dgyxdy = b[[1]] * dgyxdx \
       + b[[2]] / x * (D[b[[1]], phi] - b[[2]]) \
       + b[[3]] * dgyxdz
(* Directional derivative of b_zc along b. Since the
   cartesian zc and cylindrical Z always coincide,
   no special treatment is necessary here. *)
dgyzdy = b . Grad[b[[3]], {x, phi, z}, "Cylindrical"];

Print["Writing Dommaschk fields to file"]
Export[StringJoin["equi_ref_absB_domm.txt"],
    {"absB_domm = " <> F90Format[absB]}]
Export[StringJoin["equi_ref_normb_x_domm.txt"],
    {"normb_x_domm = " <> F90Format[b[[1]]]}]
Export[StringJoin["equi_ref_normb_y_domm.txt"],
    {"normb_y_domm = " <> F90Format[b[[2]]]}]
Export[StringJoin["equi_ref_normb_z_domm.txt"],
    {"normb_z_domm = " <> F90Format[b[[3]]]}]
Export[StringJoin["equi_ref_curl_normb_y_domm.txt"],
    {"curl_normb_y_domm = " <> F90Format[curlnormby]}]
Export[StringJoin["equi_ref_dabsBdx_domm.txt"],
    {"dabsBdx_domm = " <> F90Format[dabsBdx]}]
Export[StringJoin["equi_ref_dabsBdy_domm.txt"],
    {"dabsBdy_domm = " <> F90Format[dabsBdy]}]
Export[StringJoin["equi_ref_dabsBdz_domm.txt"],
    {"dabsBdz_domm = " <> F90Format[dabsBdz]}]
Export[StringJoin["equi_ref_dgyzdx_domm.txt"],
    {"dgyzdx_domm = " <> F90Format[dgyzdx]}]
Export[StringJoin["equi_ref_dgyxdz_domm.txt"],
    {"dgyxdz_domm = " <> F90Format[dgyxdz]}]
Export[StringJoin["equi_ref_dgyzdz_domm.txt"],
    {"dgyzdz_domm = " <> F90Format[dgyzdz]}]
Export[StringJoin["equi_ref_dgyxdx_domm.txt"],
    {"dgyxdx_domm = " <> F90Format[dgyxdx]}]
Export[StringJoin["equi_ref_dgyxdy_domm.txt"],
    {"dgyxdy_domm = " <> F90Format[dgyxdy]}]
Export[StringJoin["equi_ref_dgyzdy_domm.txt"],
    {"dgyzdy_domm = " <> F90Format[dgyzdy]}]
