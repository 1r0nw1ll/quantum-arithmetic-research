# QA Whittaker Two-Scalar-Potential Bridge Cert v1 Spec

Candidate family ID: `[507]`

Primary source: E. T. Whittaker (1904), "On an expression of the
electromagnetic field due to electrons by means of two scalar potential
functions," Proc. London Math. Soc. s2-1:367-372. DOI: 10.1112/plms/s2-1.1.367.

## Whittaker's operator (verbatim, p.370 sec.3, Phi<-F, Psi<-G)

```
dx = d^2(Phi)/dx dz + (1/c) d^2(Psi)/dy dt
dy = d^2(Phi)/dy dz - (1/c) d^2(Psi)/dx dt
dz = d^2(Phi)/dz^2 - (1/c^2) d^2(Phi)/dt^2

hx = (1/c) d^2(Phi)/dy dt - d^2(Psi)/dx dz
hy = -(1/c) d^2(Phi)/dx dt - d^2(Psi)/dy dz
hz = d^2(Psi)/dx^2 + d^2(Psi)/dy^2
```

`Phi, Psi` each independently satisfy the scalar wave equation
`grad^2(u) = (1/c^2) d^2(u)/dt^2` (Whittaker p.372); this cert does not
require that fact for `dx,dy,hx,hy,hz` divergence identities, only for the
`dz`/`div(d)_Phi` conditional gate (see below).

## Plane-wave ansatz and coefficient algebra

A formal packet is `trig(theta)` with `theta = Kx x + Ky y + Kz z - Kt t`,
where `(Kx,Ky,Kz,Kt) = k*(omega_x, omega_y, omega_z, v)` for a QA-rational
`k, v` and `omega` an exact rational unit vector supplied by registered
`[273]`. `trig` is a formal label (never evaluated); only the following
exact-Fraction mixed-partial rules are used:

```
d^2/du dw [trig(theta)] = -(Ku * Kw) * trig(theta)     (label preserved, u != t, w != t)
d^2/du dt [trig(theta)] =  (Ku * Kt) * trig(theta)     (label preserved)
d^2/dt dt [trig(theta)] = -(Kt * Kt) * trig(theta)
```

Applying these to Whittaker's operator gives twelve raw coefficients
(`{dx,dy,dz,hx,hy,hz} x {phi,psi}` channel, with `dz_psi = hz_phi = 0`
identically):

```
dx_phi = -Kx*Kz            dx_psi = (Ky*Kt)/c
dy_phi = -Ky*Kz            dy_psi = -(Kx*Kt)/c
dz_phi = Kt^2/c^2 - Kz^2    dz_psi = 0

hx_phi = (Ky*Kt)/c          hx_psi = Kx*Kz
hy_phi = -(Kx*Kt)/c         hy_psi = Ky*Kz
hz_phi = 0                  hz_psi = -(Kx^2+Ky^2)
```

## Divergence identities

```
div_d_phi = Kx*dx_phi + Ky*dy_phi + Kz*dz_phi
div_d_psi = Kx*dx_psi + Ky*dy_psi + Kz*dz_psi
div_h_phi = Kx*hx_phi + Ky*hy_phi + Kz*hz_phi
div_h_psi = Kx*hx_psi + Ky*hy_psi + Kz*hz_psi
```

Algebraically (verified exactly, not just numerically, by the validator on
every fixture):

```
div_d_psi = (Kx*Ky*Kt - Ky*Kx*Kt)/c = 0                         (always)
div_h_phi = (Kx*Ky*Kt - Ky*Kx*Kt)/c = 0                         (always)
div_h_psi = Kz*(Kx^2+Ky^2) - Kz*(Kx^2+Ky^2) = 0                 (always)
div_d_phi = Kz*(Kt^2/c^2 - (Kx^2+Ky^2+Kz^2))
          = Kz*(Kt^2/c^2 - k^2)          [since omega is a unit vector]
          = 0  when v^2 = c^2, or when Kz = 0

For nonzero-Kz packets, div_d_phi = 0 iff v^2 = c^2. The Kz = 0 exception is
intentional: a zero-z direction kills the entire Phi-channel divergence even
when the packet does not satisfy dispersion.
```

`div_d_phi` is the one identity that depends on the vacuum dispersion
relation `v^2 = c^2` — the standard EM plane-wave phase-velocity condition.
This is a genuine, checkable asymmetry between the `E`- and `H`-divergence
conditions of Whittaker's operator, not an artifact of this cert.

## Rational Encoding

Rational scalars are JSON objects `{"num": int, "den": positive int}`,
reduced, positive denominator (same convention as `[273]`/`[498]`).

`omega_packet` is `[x_num, y_num, z_num, den]`, must satisfy the `S^2`
identity `x_num^2+y_num^2+z_num^2 = den^2`, and must be an exact member of
`[273]`'s `D_m^(2)` for the fixture's declared `m in {3,5,9}`.

## Packet Schema

```
packet_id                    string, unique
omega_packet                 [x_num,y_num,z_num,den], member of [273] D_m^(2)
k, v, c                      rational objects; c != 0
wave_equation_satisfied      bool; must equal v*v == c*c exactly
coefficients                 object with the 12 COEFFICIENT_KEYS, each a rational witness
divergences                  object with div_d_phi/div_d_psi/div_h_phi/div_h_psi witnesses
```

## Gates

See `README.md` for the gate table (`WSPB_1`..`WSPB_8`, `WSPB_SCHEMA`,
`WSPB_F`).

## Non-Claims / Claim Policy

Both `non_claims` (free-text, must mention specific terms) and
`claim_policy` (typed booleans, all must be `false`) are required on every
fixture. This cert makes no claim about Maxwell derivation, electromagnetism,
physical field reconstruction, scalar-wave-energy physics, Mie scattering, or
Layer 5/6 of the Whittaker -> QA ladder.
