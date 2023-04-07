---
title: Discrete Harmonic Map
copyright: true
permalink: 1
top: 0
date: 2019-06-25 10:52:40
tags:
- Computer vision
- C++
categories:
- Computer vision
- C++
password: 
mathjax: true
---
&emsp;&emsp;Discrete Harmonic Map (DHM) parameterizes disk-like surfaces by minimizing the Dirichlet energy of the (piece-wise linear) mapping functions. It is super easy to implement (if you are familiar with cotangent-weight Laplacian and have a linear solver at hand, it should take no more than a few hours) and generally gives good results. If you never heard about it, Misha's class note  might be helpful.
# Source Code

## Cotangent Edge Weight

```C++
template<typename M>
void CHarmonicMapper<M>::_calculate_edge_weight()
{
	//compute edge length
	for (M::MeshEdgeIterator eiter(m_pMesh); !eiter.end(); ++eiter)
	{
		M::CEdge* pE = *eiter;
		M::CVertex* v1 = m_pMesh->edgeVertex1(pE);
		M::CVertex* v2 = m_pMesh->edgeVertex2(pE);
		pE->length() = (v1->point() - v2->point()).norm();
	}

	//compute corner angle
	for (M::MeshEdgeIterator eiter(m_pMesh); !eiter.end(); ++eiter)
	{
		M::CEdge* pE = *eiter;
		if (pE->boundary())
			continue;

		for (int i = 0; i < 2; ++i)
		{
			M::CHalfEdge* h = m_pMesh->edgeHalfedge(pE, i);

			double h_j = m_pMesh->halfedgeEdge((M::CHalfEdge*)h->he_next())->length();
			double h_k = m_pMesh->halfedgeEdge((M::CHalfEdge*)h->he_prev())->length();

			h->angle() = _inverse_cosine_law(h_j, h_k, pE->length());
		}
	}

	//compute edge weight
	for (M::MeshEdgeIterator eiter(m_pMesh); !eiter.end(); ++eiter)
	{
		M::CEdge* pE = *eiter;
		M::CHalfEdge* h0 = m_pMesh->edgeHalfedge(pE, 0);
		M::CHalfEdge* h1 = m_pMesh->edgeHalfedge(pE, 1);

		pE->weight() = 0.5 * (1 / tan(h0->angle() + 1 / tan(h1->angle())));
	}
}
```

## Set Boundary Condition

```c++
template<typename M>
void CHarmonicMapper<M>::_set_boundary()
{
	//get the boundary half edge loop
	std::vector<M::CLoop*> & pLs =  m_boundary.loops();
	assert( pLs.size() == 1 );
	M::CLoop * pL = pLs[0];
	std::list<M::CHalfEdge*> & pHs = pL->halfedges();
	
	//compute the total length of the boundary
	double total_length = pL->length();

	//parameterize the boundary using arc length parameter
	double current_length = 0;
	for (std::list<M::CHalfEdge*>::iterator hiter = pHs.begin(); hiter != pHs.end(); hiter++) 
	{
		M::CHalfEdge* pH = *hiter;

		M::CEdge* pE = m_pMesh->halfedgeEdge(pH);
		current_length += pE->length();
		double angle = 2 * PI / total_length * current_length;
		
		MeshLib::CPoint2* huv = new MeshLib::CPoint2(0.5 + 0.5 * cos(angle), 0.5 + 0.5 * sin(angle));
		M::CVertex* pV = m_pMesh->halfedgeTarget(pH);
		pV->huv() = *huv;
	}
}
```

## Iterative Algorithm for Harmonic Map

```c++
template<typename M>
void CHarmonicMapper<M>::_iterative_map( double epsilon )
{
	//fix the boundary
	_set_boundary();

	//move interior each vertex to its center of neighbors
	
	for (M::MeshVertexIterator viter(m_pMesh); !viter.end(); ++viter) {
		M::CVertex* pV = *viter;
		if (pV->boundary())
			continue;
		MeshLib::CPoint2* huv = new MeshLib::CPoint2(0, 0);
		pV->huv() = *huv;
	}

	while (true)
	{
		double error = -1e+10;
		//move interior each vertex to its center of neighbors
		for (M::MeshVertexIterator viter(m_pMesh); !viter.end(); ++viter) 
		{
			M::CVertex* pV = *viter;
			if (pV->boundary())
				continue;
			double total_weight = 0;
			CPoint2 huv(0, 0);
			for (M::VertexVertexIterator vviter(pV); !vviter.end(); ++vviter) 
			{
				M::CVertex* pW = *vviter;
				M::CEdge* pE = m_pMesh->vertexEdge(pV, pW);
				double weight = pE->weight();
				total_weight += weight;
				huv = huv + pW->huv() * weight;
			}
			huv = huv / total_weight;
			double _error = (pV->huv() - huv).norm();
			error = (_error > error) ? _error : error;
			pV->huv() = huv;
		}

		if (error < epsilon) break;
	}

}
```

## Direct Algorithm for Harmonic Map (Extra Credit) & Numerical Method

```c++
template<typename M>
void CHarmonicMapper<M>::_map()
{
	
	//fix the boundary
	_set_boundary();
	
	std::vector<Eigen::Triplet<double> > A_coefficients;
	std::vector<Eigen::Triplet<double> > B_coefficients;

	
	//set the matrix A
	for( M::MeshVertexIterator viter( m_pMesh ); !viter.end(); ++ viter )
	{
		M::CVertex * pV = *viter;
		if( pV->boundary() ) continue;
		int vid = pV->idx();

		double sw = 0;
		for( M::VertexVertexIterator witer( pV ); !witer.end(); ++ witer )
		{
			M::CVertex * pW = *witer;
			int wid = pW->idx();
			
			M::CEdge * e = m_pMesh->vertexEdge( pV, pW );
			double w = e->weight();

			if( pW->boundary() )
			{
				B_coefficients.push_back( Eigen::Triplet<double>(vid,wid,w) );
			}
			else
			{
				A_coefficients.push_back( Eigen::Triplet<double>(vid,wid, -w) );
			}
			sw += w;
		}
		A_coefficients.push_back( Eigen::Triplet<double>(vid,vid, sw ) );
	}


	Eigen::SparseMatrix<double> A( m_interior_vertices, m_interior_vertices );
	A.setZero();

	Eigen::SparseMatrix<double> B( m_interior_vertices, m_boundary_vertices );
	B.setZero();
	A.setFromTriplets(A_coefficients.begin(), A_coefficients.end());
	B.setFromTriplets(B_coefficients.begin(), B_coefficients.end());


	Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;
	std::cerr << "Eigen Decomposition" << std::endl;
	solver.compute(A);
	std::cerr << "Eigen Decomposition Finished" << std::endl;
	
	if( solver.info() != Eigen::Success )
	{
		std::cerr << "Waring: Eigen decomposition failed" << std::endl;
	}


	for( int k = 0; k < 2; k ++ )
	{
		Eigen::VectorXd b(m_boundary_vertices);
		//set boundary constraints b vector
		for (M::MeshVertexIterator viter(m_pMesh); !viter.end(); ++viter) {
			M::CVertex* pV = *viter;
			if (!pV->boundary())
				continue;
			int id = pV->idx();
			b(id) = pV->huv()[k];
		}

		Eigen::VectorXd c(m_interior_vertices);
		c = B * b;

		Eigen::VectorXd x = solver.solve(c);
		if( solver.info() != Eigen::Success )
		{
			std::cerr << "Waring: Eigen decomposition failed" << std::endl;
		}

		//set the images of the harmonic map to interior vertices
		for (M::MeshVertexIterator viter(m_pMesh); !viter.end(); ++viter) {
			M::CVertex* pV = *viter;
			if (pV->boundary())
				continue;
			int id = pV->idx();
			pV->huv()[k] = x(id);
		}
	}
}
```

# Algorithm

## Cotangent Edge Weight

### Compute edge length

<font size=3>Compute the edge length, suppose $e = [v_i , v_j ]$, then
$$l(e) = |v_j − v_i |.$$

<font size=3>
- use $MeshEdgeIterator$ to find all of edges on m_pMesh
- use $m_pMesh->halfedgeVertex1(Edge)$ to find the first vertex of an edge
- use $m_pMesh->halfedgeVertex2(Edge)$ to find the second vertex of an edge
- $point.norm()$ calculate the norm of the CPoint $\sqrt{x^2+y^2+z^2}$

```c++
for (M::MeshEdgeIterator eiter(m_pMesh); !eiter.end(); ++eiter)
	{
		M::CEdge* pE = *eiter;
		M::CVertex* v1 = m_pMesh->edgeVertex1(pE);
		M::CVertex* v2 = m_pMesh->edgeVertex2(pE);
		pE->length() = (v1->point() - v2->point()).norm();
	}
```

### Compute corner angle

<font size=3>
Compute the corner angles of each triangular face, suppose $[v_i , v_j , v_k ]$
is a face, with edges $e_i , e_j , e_k ,$ where e_i is against the vertex v_i . The corresponding edge lengths are $l_i , l_j$ and $l_k$ . The inversive cosine law gives

    $$ \theta_i = acos \frac {l_j^2 + l_k^2 -l_i^2} {2 l_j l_k} $$

<font size=3>
- use $MeshEdgeIterator$ to find all of edges on m_pMesh
- use $pE->boundary()$ to judge whether the edge is on the boundary
- use $m_pMesh->halfedgeEdge(halfedge)$ to attach halfedge to an edge
- use $halfedge->he\_next()$ to find the next halfedge of a halfedge
- $\_inverse\_cosine\_law(l_j,l_k,l_i)$ calculate the acos

```c++
for (M::MeshEdgeIterator eiter(m_pMesh); !eiter.end(); ++eiter)
	{
		M::CEdge* pE = *eiter;
		if (pE->boundary())
			continue;

		for (int i = 0; i < 2; ++i)
		{
			M::CHalfEdge* h = m_pMesh->edgeHalfedge(pE, i);

			double h_j = m_pMesh->halfedgeEdge((M::CHalfEdge*)h->he_next())->length();
			double h_k = m_pMesh->halfedgeEdge((M::CHalfEdge*)h->he_prev())->length();

			h->angle() = _inverse_cosine_law(h_j, h_k, pE->length());
		}
	}
```

### Compute edge weight

<font size=3>
The cotangent edge weight is as follows. Suppose edge $[v_i , v_j , v_k ] $
    and $[v_j , v_i , v_l ]$ share an edge $[v_i , v_j ]$, then
    $$w_{ij}=\frac 1 2 (cot\theta_k^{ij} + cot\theta_l^{ji})$$

<font size=3>
- use $Edge->weight()$ to get the weight of edge
- use $HalfEdge->angle()$ to get the angle of halfege
- use $1/ tan(\theta)$ to calculate $cot\theta$

```c++
for (M::MeshEdgeIterator eiter(m_pMesh); !eiter.end(); ++eiter)
	{
		M::CEdge* pE = *eiter;
		M::CHalfEdge* h0 = m_pMesh->edgeHalfedge(pE, 0);
		M::CHalfEdge* h1 = m_pMesh->edgeHalfedge(pE, 1);

		pE->weight() = 0.5 * (1 / tan(h0->angle() + 1 / tan(h1->angle())));
	}
```

## Set Boundary Condition

### Get the boundary half edge loop

<font size=3>
- use $Boundary.loops()$ to get the list of boundary loops
- use $Loop->halfedge()$ to get the the list of haledges on the current boundary loop

```C++
    std::vector<M::CLoop*> & pLs =  m_boundary.loops();
	assert( pLs.size() == 1 );
	M::CLoop * pL = pLs[0];
	std::list<M::CHalfEdge*> & pHs = pL->halfedges();
```

### Compute the total length of the boundary

<font size=3>
Suppose the boundary vertices are sorted counter-clock-wisely, 
as $\{v_0 , v_1 , · · · , v_{n−1} \}$. The total length of the boundary is given by
    $$s = \sum_{i=0}^{n-1}|v_{i+1}-v_i| $$

<font size=3>
- use $Loop->length()$ to get the length of the current boundary loop

```c++
double total_length = pL->length();
```

### Parameterize the boundary using arc length parameter

<font size=3>
    The image of $v_i \in \partial M$ is given by
    $$\begin{cases}
    \theta_i = \frac {2\pi} s \sum_{j=0}^{i-1}|v_{j+1}- v_j|\\
    \phi(v_i)=(\frac {1+cos\theta_i} 2, \frac {1+sin\theta_i} 2)
    \end{cases}$$

```c++
    double current_length = 0;
	for (std::list<M::CHalfEdge*>::iterator hiter = pHs.begin(); hiter != pHs.end(); hiter++) 
	{
		M::CHalfEdge* pH = *hiter;

		M::CEdge* pE = m_pMesh->halfedgeEdge(pH);
		current_length += pE->length();
		double angle = 2 * PI / total_length * current_length;
		
		MeshLib::CPoint2* huv = new MeshLib::CPoint2(0.5 + 0.5 * cos(angle), 0.5 + 0.5 * sin(angle));
		M::CVertex* pV = m_pMesh->halfedgeTarget(pH);
		pV->huv() = *huv;
	}
```

## Iterative Algorithm for Harmonic Map

<font size=3>
First, for each interior vertex $v_i \notin \partial M$, set $\phi(v_i ) = (0, 0)$. Second, for each interior vertex, move its image to the mass center of the images of its neighbors,
    $$c_i = \frac {\sum_j w_{ij}\phi(v_j)} {\sum_j w_ij}, \phi(v_i)\leftarrow c_i$$
    repeat this procedure, until the algorithm converges.

### For each interior vertex $v_i \notin \partial M$, set $\phi(v_i ) = (0, 0)$

<font size=3>
- use $pE->boundary()$ to judge whether the edge is on the boundary

```c++
for (M::MeshVertexIterator viter(m_pMesh); !viter.end(); ++viter) 
{
		M::CVertex* pV = *viter;
		if (pV->boundary())
			continue;
		MeshLib::CPoint2* huv = new MeshLib::CPoint2(0, 0);
		pV->huv() = *huv;
}
```

### Move interior each vertex to its center of neighbors

<font size=3>
- use $MeshVertexIterator$ to find all of edges on m_pMesh
- use $VertexVertexIterator$ to transverse all the neighboring vertices of a vertex
- use $Edge->vertexEdge(vertex0, vertex2)$ to access an edge by its two end vertices
- use $m_pMesh->weight()$ to get the weight of edge
- use $Vertex->huv()$ to get the $\phi(v_i)$ of interior vertex


```c++
while (true)
	{
		double error = -1e+10;
		for (M::MeshVertexIterator viter(m_pMesh); !viter.end(); ++viter) 
		{
			M::CVertex* pV = *viter;
			if (pV->boundary())
				continue;
			double total_weight = 0;
			CPoint2 huv(0, 0);
			for (M::VertexVertexIterator vviter(pV); !vviter.end(); ++vviter) 
			{
				M::CVertex* pW = *vviter;
				M::CEdge* pE = m_pMesh->vertexEdge(pV, pW);
				double weight = pE->weight();
				total_weight += weight;
				huv = huv + pW->huv() * weight;
			}
			huv = huv / total_weight;
			double _error = (pV->huv() - huv).norm();
			error = (_error > error) ? _error : error;
			pV->huv() = huv;
		}

		if (error < epsilon) break;
	}
```

## Direct Algorithm for Harmonic Map (Extra Credit)

<font size=3>For each interior vertex $v_i \notin \partial M$, establish one linear equation
    $$\sum_j w_{ij}(\phi(v_i)-\phi(v_j)) = 0$$
   solve this sparse linear system, the result is the harmonic map.

```c++
//set boundary constraints b vector
for (M::MeshVertexIterator viter(m_pMesh); !viter.end(); ++viter)
{
    M::CVertex* pV = *viter;
    if (!pV->boundary())
        continue;
    int id = pV->idx();
    b(id) = pV->huv()[k];
}
```

```c++
//set the images of the harmonic map to interior vertices
for (M::MeshVertexIterator viter(m_pMesh); !viter.end(); ++viter) 
{
    M::CVertex* pV = *viter;
    if (pV->boundary())
        continue;
    int id = pV->idx();
    pV->huv()[k] = x(id);
}
```

<font size=3>This assignment uses Eigen library to solve large sparse linear system. The following commands will be useful.

```c++
std::vector<Eigen::Triplet<double> > A_coefficients; 
A_coefficients.push_back( Eigen::Triplet<double>(vid,wid, -w) ); 
Eigen::SparseMatrix<double> A( m_interior_vertices, m_interior_vertices );

A.setZero();

A.setFromTriplets(A_coefficients.begin(), A_coefficients.end()); Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver; 
std::cerr << "Eigen Decomposition" << std::endl; solver.compute(A); 
std::cerr << "Eigen Decomposition Finished" << std::endl;

if( solver.info() != Eigen::Success ) 
{ 
    std::cerr << "Waring: Eigen decomposition failed" << std::endl; 
} 
Eigen::VectorXd b(m_boundary_vertices); 
Eigen::VectorXd x = solver.solve(c); 
if( solver.info() != Eigen::Success ) 
{ 
    std::cerr << "Waring: Eigen decomposition failed" << std::endl; 
}
```

# Result

<font size=3>In this section, we show the result of harmonic mapping separately.
- Direct Algorithm for Harmonic Map(Fig. 3.1 & Fig. 3.2)   
    <br/>
- Iterative Algorithm for Harmonic Map(Fig. 3.3)

<font size=3>
<table>
    <tr>
        <td><img src='./result/Alex_circle_map.png'></td>
        <td><img src='./result/Alex_go_map.png'></td>
        <td><img src='./result/Alex_uv_map.png'></td>
    </tr>   
</table>
    <center>Fig. 3.1. Harmonic Map On Alex 

<font size=3>
<table>
    <tr>
        <td><img src='./result/sophie_circle_map.png'></td>
        <td><img src='./result/sophie_go_map.png'></td>
        <td><img src='./result/sophie_uv_map.png'></td>
    </tr>   
</table>
    <center>Fig. 3.2. Harmonic Map On Sophie

<font size=3>
<table>
    <tr>
        <td><img src='./result/sophie_circle_iter_map.png'></td>
        <td><img src='./result/sophie_go_iter_map.png'></td>
        <td><img src='./result/sophie_uv_iter_map.png'></td>
    </tr>   
</table>
    <center>Fig. 3.3. Iterative Harmonic Map On Sophie

<font size=3>
Data from sophie.uv.m

```
Vertex 54475 103.073 50.793 26.4526 {rgb=(0.117956 0.117956 0.117956) uv=(0.999907 0.490347)}
Vertex 54480 107.076 50.7952 27.7994 {rgb=(0.151779 0.151779 0.151779) uv=(0.999013 0.468605)}
Vertex 54487 108.817 50.7937 28.1584 {rgb=(0.156541 0.156541 0.156541) uv=(0.998355 0.459469)}
Vertex 54496 114.037 50.795 29.4016 {rgb=(0.174857 0.174857 0.174857) uv=(0.995353 0.431987)}
Vertex 54505 117.28 50.7941 29.6458 {rgb=(0.16433 0.16433 0.16433) uv=(0.992795 0.415427)}
```

<font size=3>Data generated by Direct Algorithm for Harmonic Map

```
Vertex 54475  0.999907 0.490347 0 {rgb=(0.117956 0.117956 0.117956)}
Vertex 54480  0.999013 0.468605 0 {rgb=(0.151779 0.151779 0.151779)}
Vertex 54487  0.998355 0.459469 0 {rgb=(0.156541 0.156541 0.156541)}
Vertex 54496  0.995353 0.431987 0 {rgb=(0.174857 0.174857 0.174857)}
Vertex 54505  0.992795 0.415427 0 {rgb=(0.16433 0.16433 0.16433)}
```

<font size=3>Data generated by Iterative Algorithm for Harmonic Map

```
Vertex 54475  0.999907 0.490347 0 {rgb=(0.117956 0.117956 0.117956)}
Vertex 54480  0.999013 0.468605 0 {rgb=(0.151779 0.151779 0.151779)}
Vertex 54487  0.998355 0.459469 0 {rgb=(0.156541 0.156541 0.156541)}
Vertex 54496  0.995353 0.431987 0 {rgb=(0.174857 0.174857 0.174857)}
Vertex 54505  0.992795 0.415427 0 {rgb=(0.16433 0.16433 0.16433)}
```
