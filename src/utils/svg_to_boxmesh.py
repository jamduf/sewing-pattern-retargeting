#!/usr/bin/env python3

import argparse
import math
import struct
import tarfile
import xml.etree.ElementTree as ET
import re

import numpy as np
import mapbox_earcut as earcut


class Polygon:
    def __init__(self, outer, label):
        self.outer = outer
        self.label = label


def arc_to_poly(x0, y0, rx, ry, phi, large_arc, sweep, x1, y1, tol=0.75):
    phi = math.radians(phi)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    dx = (x0 - x1) / 2
    dy = (y0 - y1) / 2

    x1p = cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy

    rx = abs(rx)
    ry = abs(ry)

    lam = (x1p**2)/(rx**2) + (y1p**2)/(ry**2)
    if lam > 1:
        s = math.sqrt(lam)
        rx *= s
        ry *= s

    rx2 = rx*rx
    ry2 = ry*ry
    x1p2 = x1p*x1p
    y1p2 = y1p*y1p

    sign = -1 if large_arc == sweep else 1
    sq = max(0, (rx2*ry2 - rx2*y1p2 - ry2*x1p2)/(rx2*y1p2 + ry2*x1p2))
    coef = sign * math.sqrt(sq)

    cxp = coef * (rx * y1p)/ry
    cyp = coef * (-ry * x1p)/rx

    cx = cos_phi * cxp - sin_phi * cyp + (x0 + x1)/2
    cy = sin_phi * cxp + cos_phi * cyp + (y0 + y1)/2

    theta1 = math.atan2((y1p - cyp)/ry, (x1p - cxp)/rx)
    dtheta = math.atan2((-y1p - cyp)/ry, (-x1p - cxp)/rx) - theta1

    if not sweep and dtheta > 0:
        dtheta -= 2*math.pi
    elif sweep and dtheta < 0:
        dtheta += 2*math.pi

    n = max(4, int(abs(dtheta)*max(rx,ry)/tol))

    pts=[]
    for i in range(1,n+1):
        t=theta1+dtheta*i/n
        x=cx+rx*math.cos(t)*cos_phi - ry*math.sin(t)*sin_phi
        y=cy+rx*math.cos(t)*sin_phi + ry*math.sin(t)*cos_phi
        pts.append((x,y))

    return pts


def parse_svg(svg_bytes):
    root = ET.fromstring(svg_bytes)
    polys=[]

    for idx,el in enumerate(root.findall(".//{http://www.w3.org/2000/svg}path")):
        d=el.attrib.get("d")
        if not d:
            continue

        tokens=re.findall(r"[MLAQZmlaqz]|-?\d*\.?\d+(?:e[-+]?\d+)?",d)

        i=0
        x=y=0
        pts=[]

        while i<len(tokens):
            cmd=tokens[i]
            i+=1

            if cmd.upper()=="M":
                x=float(tokens[i])
                y=float(tokens[i+1])
                i+=2
                pts.append((x,y))

            elif cmd.upper()=="L":
                x=float(tokens[i])
                y=float(tokens[i+1])
                i+=2
                pts.append((x,y))

            elif cmd.upper()=="A":
                rx=float(tokens[i])
                ry=float(tokens[i+1])
                phi=float(tokens[i+2])
                large=int(tokens[i+3])
                sweep=int(tokens[i+4])
                x1=float(tokens[i+5])
                y1=float(tokens[i+6])
                i+=7

                arcpts=arc_to_poly(x,y,rx,ry,phi,large,sweep,x1,y1)
                pts.extend(arcpts)
                x,y=x1,y1

            elif cmd.upper()=="Z":
                break

        if len(pts)>=3:
            polys.append(Polygon(pts,f"panel_{idx}"))

    return polys


def triangulate(poly):
    coords = np.array(poly.outer, dtype=np.float64)

    # mapbox_earcut 2.0.0 expects ring_end_indices (end index of each ring)
    ring_ends = np.array([coords.shape[0]], dtype=np.uint32)

    tri = earcut.triangulate_float64(coords, ring_ends)
    faces = tri.reshape(-1, 3)
    return coords, faces

def build(svg_bytes):
    polys=parse_svg(svg_bytes)

    V=[]
    UV=[]
    F=[]
    labels=[]

    offset=0

    for p in polys:

        v2,f=triangulate(p)

        v3=np.zeros((len(v2),3),dtype=np.float32)
        v3[:,:2]=v2.astype(np.float32)

        xmin,xmax=v2[:,0].min(),v2[:,0].max()
        ymin,ymax=v2[:,1].min(),v2[:,1].max()

        s=(v2[:,0]-xmin)/(xmax-xmin+1e-9)
        t=(v2[:,1]-ymin)/(ymax-ymin+1e-9)

        uv=np.stack([s,t],axis=1).astype(np.float64)

        V.append(v3)
        UV.append(uv)
        F.append(f+offset)

        labels.extend([p.label]*len(v3))

        offset+=len(v3)

    return np.vstack(V),np.vstack(UV),np.vstack(F),labels


def write_ply(path,V,UV,F):

    with open(path,"wb") as f:

        header=f"""ply
format binary_little_endian 1.0
comment https://github.com/mikedh/trimesh
element vertex {len(V)}
property float x
property float y
property float z
property double s
property double t
element face {len(F)}
property list uchar int vertex_indices
end_header
"""

        f.write(header.encode())

        for (x,y,z),(s,t) in zip(V,UV):
            f.write(struct.pack("<fffdd",x,y,z,s,t))

        for a,b,c in F:
            f.write(struct.pack("<Biii",3,a,b,c))


def main():

    parser=argparse.ArgumentParser()

    parser.add_argument("--tar")
    parser.add_argument("--member")

    parser.add_argument("--out_ply",required=True)
    parser.add_argument("--out_seg")

    args=parser.parse_args()

    with tarfile.open(args.tar) as tf:
        svg=tf.extractfile(args.member).read()

    V,UV,F,labels=build(svg)

    write_ply(args.out_ply,V,UV,F)

    if args.out_seg:
        with open(args.out_seg,"w") as f:
            for l in labels:
                f.write(l+"\n")

    print("SUCCESS:",args.out_ply)


if __name__=="__main__":
    main()
