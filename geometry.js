// Minkowski Pentahedron: geometry and solver.
// Pure computation, no drawing or DOM code. Exposes the global `Pentahedron` in a browser,
// and the same object through module.exports under Node.
(function (root) {
  "use strict";

  // ---------- vector algebra ----------
  const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  const sub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
  const norm = a => Math.sqrt(dot(a, a));
  const meanPt = pts => [0, 1, 2].map(t => pts.reduce((s, x) => s + x[t], 0) / pts.length);

  // ---------- the five vectors ----------
  // Parameters are in units of π: the angle used is π times the parameter value.
  // v1 = (0,0,-1), v2 = (-sin a, 0, cos a), v3 = (sin b cos c, sin b sin c, cos b),
  // v4 = (sin d cos e, sin d sin e, cos d), v5 = -(v1+v2+v3+v4).
  function vectors(p) {
    const a = Math.PI*p.a, b = Math.PI*p.b, c = Math.PI*p.c, d = Math.PI*p.d, e = Math.PI*p.e;
    const v = [
      [0, 0, -1],
      [-Math.sin(a), 0, Math.cos(a)],
      [Math.sin(b)*Math.cos(c), Math.sin(b)*Math.sin(c), Math.cos(b)],
      [Math.sin(d)*Math.cos(e), Math.sin(d)*Math.sin(e), Math.cos(d)]
    ];
    v.push([0, 1, 2].map(t => -(v[0][t] + v[1][t] + v[2][t] + v[3][t])));
    return v;
  }

  // ---------- polytope from support numbers ----------
  // Polytope {x : u_i . x <= h_i}: vertices, ordered face polygons, face areas, volume.
  function polytope(U, h) {
    const n = U.length, verts = [];
    const scale = Math.max(1, ...h.map(Math.abs));
    for (let i = 0; i < n; i++) for (let j = i+1; j < n; j++) for (let k = j+1; k < n; k++) {
      const c1 = cross(U[j], U[k]), det = dot(U[i], c1);
      if (Math.abs(det) < 1e-12) continue;
      const c2 = cross(U[k], U[i]), c3 = cross(U[i], U[j]);
      const x = [0, 1, 2].map(t => (h[i]*c1[t] + h[j]*c2[t] + h[k]*c3[t]) / det);
      let inside = true;
      for (let m = 0; m < n; m++) if (dot(U[m], x) > h[m] + 1e-9*scale) { inside = false; break; }
      if (!inside || verts.some(v => norm(sub(v, x)) < 1e-7*scale)) continue;
      verts.push(x);
    }
    const faces = [], areas = [];
    let vol = 0;
    for (let i = 0; i < n; i++) {
      const on = verts.filter(v => Math.abs(dot(U[i], v) - h[i]) < 1e-7*scale);
      if (on.length < 3) { faces.push(on); areas.push(0); continue; }
      const c = meanPt(on);
      let e1 = sub(on[0], c); const l1 = norm(e1) || 1; e1 = e1.map(t => t / l1);
      const e2 = cross(U[i], e1);
      const ang = p => Math.atan2(dot(sub(p, c), e2), dot(sub(p, c), e1));
      on.sort((p, q) => ang(p) - ang(q));
      let A = 0;
      for (let t = 0; t < on.length; t++) A += dot(U[i], cross(sub(on[t], c), sub(on[(t+1) % on.length], c))) / 2;
      faces.push(on); areas.push(Math.max(A, 0)); vol += h[i] * A / 3;
    }
    return { verts, faces, areas, vol };
  }

  // ---------- centroids ----------
  // Centre of area of a convex polygon (vertices in order), by fan triangulation.
  function faceCentroid(poly) {
    const o = poly[0], acc = [0, 0, 0];
    let A = 0;
    for (let t = 1; t < poly.length - 1; t++) {
      const w = norm(cross(sub(poly[t], o), sub(poly[t+1], o)));
      A += w;
      for (let k = 0; k < 3; k++) acc[k] += w * (o[k] + poly[t][k] + poly[t+1][k]) / 3;
    }
    return A > 1e-14 ? acc.map(x => x / A) : meanPt(poly);
  }

  // Centre of mass of a convex polytope, by tetrahedra from an interior point.
  function solidCentroid(P) {
    const r = meanPt(P.verts), acc = [0, 0, 0];
    let V = 0;
    P.faces.forEach(f => {
      if (f.length < 3) return;
      for (let t = 1; t < f.length - 1; t++) {
        const w = Math.abs(dot(sub(f[0], r), cross(sub(f[t], r), sub(f[t+1], r))));
        V += w;
        for (let k = 0; k < 3; k++) acc[k] += w * (r[k] + f[0][k] + f[t][k] + f[t+1][k]) / 4;
      }
    });
    return V > 1e-14 ? acc.map(x => x / V) : r;
  }

  // ---------- Minkowski solver ----------
  // Maximize log V(h) - 3 log(sum A_i h_i) (concave, scale invariant); the optimum has areas proportional to A.
  // Then rescale so the face areas equal A.
  function solveMinkowski(U, A, h0) {
    const n = U.length;
    const S = hh => hh.reduce((s, x, i) => s + A[i]*x, 0);
    const f = hh => { const P = polytope(U, hh); return { v: P.vol > 0 ? Math.log(P.vol) - 3*Math.log(S(hh)) : -Infinity, P }; };
    let h = (h0 && h0.length === n) ? h0.slice() : new Array(n).fill(1);
    let cur = f(h);
    if (!isFinite(cur.v)) { h = new Array(n).fill(1); cur = f(h); }
    if (!isFinite(cur.v)) return null;
    let step = 1;
    for (let it = 0; it < 500; it++) {
      const P = cur.P, s = S(h);
      const ratio = P.areas.map((F, i) => F / A[i]);
      const mean = ratio.reduce((x, y) => x + y, 0) / n;
      if (Math.max(...ratio.map(r => Math.abs(r/mean - 1))) < 1e-10) break;
      const g = P.areas.map((F, i) => F / P.vol - 3*A[i] / s);
      let moved = false;
      for (let tries = 0; tries < 40; tries++) {
        const hn = h.map((x, i) => x + step*g[i]);
        const nx = f(hn);
        if (nx.v > cur.v) { const k = 1 / S(hn); h = hn.map(x => x*k); cur = f(h); step *= 2; moved = true; break; }
        step /= 4;
      }
      if (!moved) break;
    }
    const tot = cur.P.areas.reduce((x, y) => x + y, 0) / A.reduce((x, y) => x + y, 0);
    if (!(tot > 0)) return null;
    const k = 1 / Math.sqrt(tot);
    h = h.map(x => x*k);
    return { h, P: polytope(U, h) };
  }

  // ---------- shape classification ----------
  // A prism with exactly one edge shorter than this is reported as a square pyramid.
  const PYRAMID_TOL = 0.01;
  // Polyhedra wider than this are treated as degenerate (normals nearly coplanar).
  const MAX_DIAMETER = 30;

  function shapeName(nFaces, nVerts) {
    if (nFaces === 5 && nVerts === 6) return "triangular prism";
    if (nFaces === 5 && nVerts === 5) return "square pyramid";
    if (nFaces === 4 && nVerts === 4) return "tetrahedron";
    return nFaces + " faces";
  }

  // ---------- full analysis for one parameter set ----------
  // params: {a, b, c, d, e} in units of π.
  // warm: the `warm` field of a previous result, used to start the solver nearby (optional).
  // Returns the vectors, how they group into faces, and (when ok) the polyhedron centred on its centre of mass.
  function analyze(params, warm) {
    const v = vectors(params);
    const len = v.map(norm);

    // Group vectors by direction: parallel normals belong to the same face (areas add).
    const groups = [];      // {u, A, members}
    const groupOf = [];
    for (let i = 0; i < 5; i++) {
      if (len[i] < 1e-9) { groupOf.push(-1); continue; }
      const u = v[i].map(t => t / len[i]);
      let g = groups.findIndex(G => dot(G.u, u) > 1 - 1e-9);
      if (g < 0) { groups.push({ u, A: 0, members: [] }); g = groups.length - 1; }
      groups[g].A += len[i]; groups[g].members.push(i);
      groupOf.push(g);
    }
    const U = groups.map(G => G.u), A = groups.map(G => G.A);

    // Spread of the normals: the largest |det| over triples; zero means coplanar.
    let spread = 0;
    for (let i = 0; i < U.length; i++) for (let j = i+1; j < U.length; j++) for (let k = j+1; k < U.length; k++)
      spread = Math.max(spread, Math.abs(dot(U[i], cross(U[j], U[k]))));

    const out = { v, len, groups, groupOf, ok: false, problem: "", warm: warm || null, faces: [] };
    let sol = null;
    if (U.length < 4 || spread < 1e-6) {
      out.problem = "The normals are coplanar, so no bounded polyhedron has them as face normals.";
    } else {
      const gk = groups.map(G => G.members.join("")).join("|");
      sol = solveMinkowski(U, A, warm && warm.key === gk ? warm.h : null);
      if (sol) out.warm = { h: sol.h, key: gk };
      else out.problem = "The solver did not converge for these parameters.";
    }
    if (!sol) return out;

    const P = sol.P;
    // Centre on the centre of mass, which moves continuously when an edge collapses (the vertex average does not).
    const cen = solidCentroid(P);
    const diam = Math.max(...P.verts.map(x => norm(sub(x, cen)))) * 2;
    const err = Math.max(...P.areas.map((F, g) => Math.abs(F - A[g])));
    if (diam > MAX_DIAMETER) {
      out.problem = "The normals are nearly coplanar, so the polyhedron is extremely long (diameter " + diam.toFixed(0) + "). Move a slider to bring it back.";
      return out;
    }
    if (err > 1e-4) {
      out.problem = "The solver did not converge for these parameters.";
      return out;
    }

    // Edges are consecutive vertex pairs around each face polygon; each edge is seen twice.
    const edgeLens = [];
    P.faces.forEach(f => { if (f.length >= 3) for (let t = 0; t < f.length; t++) edgeLens.push(norm(sub(f[t], f[(t + 1) % f.length]))); });
    // A square pyramid is a prism with exactly one edge collapsed to a point.
    // Requiring a single short edge keeps a prism with a tiny triangular face (|v5| near 0) from counting.
    const shortEdges = edgeLens.filter(l => l < PYRAMID_TOL).length / 2;
    const pyramid = groups.length === 5 && (P.verts.length === 5 || (P.verts.length === 6 && shortEdges === 1));

    out.ok = true;
    out.h = sol.h;
    out.vol = P.vol;
    out.err = err;
    out.diam = diam;
    out.pyramid = pyramid;
    out.nVerts = pyramid ? 5 : P.verts.length;
    out.nEdges = out.nVerts + groups.length - 2;
    out.minEdge = pyramid ? 0 : Math.min(...edgeLens);
    out.shape = shapeName(groups.length, out.nVerts);

    // One entry per input vector. A face shared by parallel vectors is listed under the first of them.
    for (let i = 0; i < 5; i++) {
      const g = groupOf[i];
      const poly = g >= 0 ? P.faces[g].map(x => sub(x, cen)) : [];
      const hasFace = poly.length >= 3;
      out.faces.push({
        group: g,
        primary: hasFace && groups[g].members[0] === i,
        poly,
        centroid: hasFace ? faceCentroid(poly) : null,
        normal: g >= 0 ? groups[g].u : null
      });
    }
    return out;
  }

  const api = {
    dot, cross, sub, norm,
    vectors, polytope, faceCentroid, solidCentroid, solveMinkowski,
    shapeName, analyze,
    PYRAMID_TOL, MAX_DIAMETER
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Pentahedron = api;
})(typeof window !== "undefined" ? window : this);
