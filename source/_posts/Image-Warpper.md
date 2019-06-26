---
title: Image Warpper
copyright: true
permalink: 1
top: 0
date: 2019-06-23 10:16:32
tags:
- Computer vision
- C++
categories:
- Computer vision
- C++
password:
mathjax: true
---

## Bezier Curve Evaluation Algorithm

<font size=3>A bézier line is an arc that is strictly based on a set number of points instead of on an ellipse. A bézier curve uses at least four points to draw on.  
     Bézier curves can be combined to form a Bézier spline, or generalized to higher dimensions to form Bézier surfaces

### Linear Bézier curves
<font size=3>Given distinct points $P_0$ and $P_1$, a linear Bézier curve is simply a straight line between those two points. The curve is given by
$$B ( t ) = P_0 + t ( P_1-P_0) = (1 - t)P_0 + tP_1 , 0 \leq t \leq 1$$
<font size=3>
The coeficients, $b_i$ are the control points or *Bézier points* and together with the basis function $B_{i,n}(t)$determine the shape of the curve. Lines drawn between consecutive control points of the curve form the *control polygon*. A cubic Bezier curve together with its control polyg`on is shown in Fig. 1.1. Bezier curves have the following properties:
- Geometry invariance property: Partition of unity property of the Bernstein polynomial assures the invariance of the shape of the Bézier curve under translation and rotation of its control points

![avatar](./Bezier curve.png)
<font size=3><center>Fig. 1.1. 

<font size=3>$Slerp(p, q, t)$, linear interplate two control points $p$ and $q$,

```c++
inline CPoint CBezier::lerp( CPoint & p, CPoint & q, double t )
{
	CPoint r = p * ( 1- t) + q * t;
	return r;
};
```

<font size=3>$deBoor(C0, C1, C2, C3, t)$, compute a point on the Bezier curve at parametert

```c++
inline CPoint CBezier::deBoor( CPoint & c0, CPoint & c1, CPoint & c2, CPoint & c3, double t )
{
	//Modify this procedure 
	CPoint C01 = lerp( c0, c1, t );
	CPoint C12 = lerp( c1, c2, t ); 
	CPoint C23 = lerp( c2, c3, t );
	CPoint C012 = lerp( C01, C12, t );
	CPoint C123 = lerp( C12, C23, t );
	CPoint C0123 = lerp( C012, C123, t );
	return C0123;
};
```

## Bezier Surface Evaluation Algorithm

<font size=3>
Given a control net $C[4][4]$, and $2D$ parameters $uv$,
evaluate the point on the Bezier surface constructed from the control net at the parameters $uv$,
the algorithm evaluate the point p[k] on four Bezier curves controlled by points 
m_control_net$[k][0]$, m_control_net$[k][1]$, m_control_net$[k][2]$, m_control_net$[k][3]$,
then evaluate the point on the Bezier curve controlled by $D[0], D[1], D[2], uv[3]$.

```c++
inline CPoint CBezier::evaluate( CPoint2 uv )
{
	//Modify this procedure
	CPoint D[4];
	for (int k = 0; k < 4; k++)
	{
		D[k] = deBoor(m_control_net[k][0], m_control_net[k][1], m_control_net[k][2], m_control_net[k][3], uv[0]);
	}
	CPoint r = deBoor( D[0], D[1], D[2], D[3], uv[1] );
	return r;
};
```

## Image Warpping

<font size=3>
The image is embedded in the unit square, the unit square is deformed by a nonlinear mapping $\phi :S \rightarrow \Re^2 $, where $S$ is the unit square. The mapping is modeled as Bezier surface, controlled by the control net.

$$\phi(u, v) = \sum_{i=0}^3\sum_{j=0}^3B_3^i(u)B_3^j(v)C_{ij},$$
where $\{C_{ij} \}$ form the control net,$B_3^i$ is the Bernstein polynomial,

$$B_3^i(t) = C_3^i(1 − t)^i t^{3−i} ,$$

## Summary

### Result

<font size=3>
<table>
    <tr>
        <td><img src='./result/brad_pitt.jpg'></td>
        <td><img src='./result/brad_pitt1.jpg'></td>
    </tr>   
</table>
    <center>Fig. 4.1. Image Warpper On Brad Pitt
<table>        
    <tr>
        <td><img src='./result/Lake.jpg'></td>
        <td><img src='./result/Lake1.jpg'></td>
    </tr>   
</table>    
    <center>Fig. 4.2. Image Warpper On Lake
<table>        
    <tr>
        <td><img src='./result/Wu.jpg'></td>
        <td><img src='./result/Wu1.jpg'></td>
    </tr>   
</table>    
     <center>Fig. 4.3. Image Warpper On Wu

### Explain your algorithm for each requirement

<font size=3>
1. The first algorithm generates Bezier Line by $Slerp(p, q, t)$ and $deBoor(C0, C1, C2, C3, t)$  

2. The second algorithm generates Bezier Surface by iterating $deBoor(C0, C1, C2, C3, t)$

