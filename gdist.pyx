# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
This module defines a Cython wrapper for the geodesic distance C++ library.
The algorithm (implemented in C++) extends Mitchell, Mount and Papadimitriou 
(1987) and was written by Danil Kirsanov (http://code.google.com/p/geodesic/)
    
The interface definitions and wrapper functions are written in Cython syntax 
(http://cython.org/) and provide an API for some of the classes, functions and 
constants useful for calculating the geodesic distance. 

To compile, and build gdist.so using Cython::
    
    python setup.py build_ext --inplace

.. moduleauthor:: Gaurav Malhotra <Gaurav@tvb.invalid>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

#For compatible datatypes
import numpy
cimport numpy

# For csc_matrix returned by local_gdist_matrix()
import scipy.sparse

# Pre-cdef'd containers from the C++ standard library   
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdio cimport printf

from cython.parallel import prange, parallel, threadid

################################################################################
############ Defininitions to access the C++ geodesic distance library #########
################################################################################
cdef extern from "geodesic_mesh_elements.h" namespace "geodesic":
    cdef cppclass Vertex:
        Vertex()

cdef extern from "geodesic_mesh_elements.h" namespace "geodesic":
    cdef cppclass SurfacePoint nogil:
        SurfacePoint()
        SurfacePoint(Vertex*)
        double& x()
        double& y()
        double& z()

cdef extern from "geodesic_mesh.h" namespace "geodesic":
    cdef cppclass Mesh nogil:
        Mesh()
        void initialize_mesh_data(vector[double]&, vector[unsigned]&)
        vector[Vertex]& vertices()

cdef extern from "geodesic_algorithm_exact.h" namespace "geodesic":
    cdef cppclass GeodesicAlgorithmExact nogil:
        GeodesicAlgorithmExact(Mesh*)
        void propagate(vector[SurfacePoint]&, double, vector[SurfacePoint]*)
        unsigned best_source(SurfacePoint&, double&)

cdef extern from "geodesic_algorithm_subdivision.h" namespace "geodesic":
    cdef cppclass GeodesicAlgorithmSubdivision nogil:
        GeodesicAlgorithmSubdivision(Mesh*, unsigned)
        void propagate(vector[SurfacePoint]&, double, vector[SurfacePoint]*)
        unsigned best_source(SurfacePoint&, double&)

cdef extern from "geodesic_algorithm_base.h" namespace "geodesic":
    cdef cppclass GeodesicAlgorithmBase nogil:
        GeodesicAlgorithmBase()
        void propagate(vector[SurfacePoint]&, double, vector[SurfacePoint]*)
        unsigned best_source(SurfacePoint&, double&)


cdef extern from "geodesic_constants_and_simple_functions.h" namespace "geodesic":
    double GEODESIC_INF
################################################################################

# Typedefintions for easier declaration of c++ vectors
ctypedef vector[SurfacePoint] surf_vec
ctypedef vector[surf_vec] surf_vec_vec

cdef class GeodesicAlgorithm:

    cdef Mesh amesh
    cdef bool approximate
    cdef unsigned subdivision_level

    def __cinit__(self, numpy.ndarray[numpy.float64_t, ndim=2] vertices,
                  numpy.ndarray[numpy.int32_t, ndim=2] triangles,
                  bool approximate=False, unsigned subdivision_level=0):
        """Python wrapper for the geodesic algorithm class. Initialization of the mesh
        is only done once, as opposed to every call of compute_gdist.
        Args:
            ``vertices``: defines x,y,z coordinates of the mesh's vertices.
            ``triangles``: defines faces of the mesh as index triplets into vertices.
            ``subdivision_level``: 0: dijkstra, inf:exact solution
            ``approximate``: use approximate algorithm?
        """
        # Define C++ vectors to contain the mesh surface components.
        cdef vector[double] points
        cdef vector[unsigned] faces
    
        # Map numpy array of mesh "vertices" to C++ vector of mesh "points" 
        cdef numpy.float64_t coord
        for coord in vertices.flatten():
            points.push_back(coord)
    
        # Map numpy array of mesh "triangles" to C++ vector of mesh "faces" 
        cdef numpy.int32_t indx
        for indx in triangles.flatten():
            faces.push_back(indx)
        
        # Initialize the mesh object
        self.amesh.initialize_mesh_data(points, faces)

        # set the other parameters
        self.approximate = approximate
        self.subdivision_level = subdivision_level 

    def compute_gdist_pairs(self, numpy.ndarray[numpy.int32_t, ndim=1] source_indices,
                            numpy.ndarray[numpy.int32_t, ndim=1] target_indices = None, 
                            numpy.int32_t num_threads = 1):
        """
        Pairwise computation of geodesic distances from source_indices to target_indices. 
        Runs in parallel, with num_threads. 
        Args:
            ``source_indices``: Index of the source on the mesh
            ``target_indices``: Index of the target on the mesh (same length as source_indices is required)
            ``num_threads``: Number of threads to speed up the computation
        Returns:
            numpy.ndarray((len(target_indices),)) of geodesic distances between the point pairs.
        """
        assert len(source_indices) == len(target_indices)

        #define variables
        cdef Py_ssize_t num_points = len(target_indices)
        cdef numpy.ndarray[numpy.float64_t, ndim=1] distances = numpy.zeros((num_points, ), dtype=numpy.float64)
        cdef surf_vec el = surf_vec(1)
        cdef surf_vec_vec sources = surf_vec_vec(num_threads, el) #thread local variable
        cdef surf_vec_vec targets = surf_vec_vec(num_threads, el) #thread local variable

        cdef Py_ssize_t k
        cdef GeodesicAlgorithmBase *algorithm
        cdef Py_ssize_t thread_id = -1
        with nogil, parallel(num_threads=num_threads):
            # create algorithm object
            if not self.approximate:
                algorithm = <GeodesicAlgorithmBase*> new GeodesicAlgorithmExact(&self.amesh)
            else:
                algorithm = <GeodesicAlgorithmBase*> new GeodesicAlgorithmSubdivision(&self.amesh, self.subdivision_level)
            # calculate distances in parallel
            for k in prange(source_indices.shape[0]):
                thread_id = threadid()
                #printf("K %d Thread %d \n", (k, thread_id))
                sources[thread_id][0] = SurfacePoint(&self.amesh.vertices()[source_indices[k]]) 
                targets[thread_id][0] = SurfacePoint(&self.amesh.vertices()[target_indices[k]]) 
    
                #Propogate to the specified targets
                algorithm.propagate(sources[thread_id], GEODESIC_INF, &targets[thread_id])
                algorithm.best_source(targets[thread_id][0], distances[k])
            del algorithm 
        distances[distances==GEODESIC_INF] = numpy.inf
        return distances

    def compute_gdist(self, numpy.ndarray[numpy.int32_t, ndim=1] source_indices = None,
                      numpy.ndarray[numpy.int32_t, ndim=1] target_indices = None,
                      double max_distance = GEODESIC_INF):
        """
        This is the wrapper function for computing geodesic distance between a set 
        of sources and targets on a mesh surface. This function takes the following arguments:
            ``source_indices``: Index of the source on the mesh.
            ``target_indices``: Index of the targets on the mesh.
            ``max_distance``: Maximal distance to which the geodesic distance is 
                calculated when propagating to the entire mesh
        and returns a numpy.ndarray((len(target_indices), ), dtype=numpy.float64) 
        specifying the shortest distance to the target vertices from the nearest 
        source vertex on the mesh. If no target_indices are provided, all vertices 
        of the mesh are considered as targets, however, in this case, specifying 
        max_distance will limit the targets to those vertices within max_distance of
        a source.
    
        """
        # Define and create object for algorithm on that mesh
        cdef GeodesicAlgorithmBase *algorithm
        if not self.approximate:
            algorithm = <GeodesicAlgorithmBase*> new GeodesicAlgorithmExact(&self.amesh)
        else:
            algorithm = <GeodesicAlgorithmBase*> new GeodesicAlgorithmSubdivision(&self.amesh, self.subdivision_level)

        # Define a C++ vector for the source vertices
        cdef vector[SurfacePoint] all_sources
    
        # Define a C++ vector for the target vertices
        cdef vector[SurfacePoint] stop_points
    
        # Define a NumPy array to hold the results
        cdef numpy.ndarray[numpy.float64_t, ndim=1] distances

        cdef Py_ssize_t  k
    
        if source_indices is None: #Default to 0
            source_indices = numpy.arange(0, dtype=numpy.int32)
        # Map the NumPy sources of targets to a C++ vector
        for k in source_indices:
            all_sources.push_back(SurfacePoint(&self.amesh.vertices()[k]))

        if target_indices is None:
            #Propagte purely on max_distance
            algorithm.propagate(all_sources, max_distance, NULL)
            #Make all vertices targets, NOTE: this is inefficient in the best_source 
            #                                 step below, but not sure how to avoid.
            target_indices = numpy.arange(self.amesh.vertices().size(), dtype=numpy.int32)
            # Map the NumPy array of targets to a C++ vector
            for k in target_indices:
                stop_points.push_back(SurfacePoint(&self.amesh.vertices()[k]))
        else:
            # Map the NumPy array of targets to a C++ vector
            for k in target_indices:
                stop_points.push_back(SurfacePoint(&self.amesh.vertices()[k]))
            #Propogate to the specified targets
            algorithm.propagate(all_sources, max_distance, &stop_points)
    
        #
        distances = numpy.zeros((len(target_indices), ), dtype=numpy.float64)
        for k in range(stop_points.size()):
            algorithm.best_source(stop_points[k], distances[k])
    
        distances[distances==GEODESIC_INF] = numpy.inf
        del algorithm
        return distances
   

def compute_gdist(numpy.ndarray[numpy.float64_t, ndim=2] vertices,
                  numpy.ndarray[numpy.int32_t, ndim=2] triangles,
                  numpy.ndarray[numpy.int32_t, ndim=1] source_indices = None,
                  numpy.ndarray[numpy.int32_t, ndim=1] target_indices = None,
                  double max_distance = GEODESIC_INF,
                  unsigned subdivision_level = 0,
                  bool approximate = False
                  ):
    """
    This is the wrapper function for computing geodesic distance between a set 
    of sources and targets on a mesh surface. This function accepts five 
    arguments:
        ``vertices``: defines x,y,z coordinates of the mesh's vertices.
        ``triangles``: defines faces of the mesh as index triplets into vertices.
        ``source_indices``: Index of the source on the mesh.
        ``target_indices``: Index of the targets on the mesh.
        ``max_distance``: Maximal distance to which the geodesic distance is 
            calculated when propagating to the entire mesh
        ``subdivision_level``: 0: dijkstra, inf:exact solution
        ``approximate``: use approximate algorithm?
    and returns a numpy.ndarray((len(target_indices), ), dtype=numpy.float64) 
    specifying the shortest distance to the target vertices from the nearest 
    source vertex on the mesh. If no target_indices are provided, all vertices 
    of the mesh are considered as targets, however, in this case, specifying 
    max_distance will limit the targets to those vertices within max_distance of
    a source.
    
    NOTE: This is the function to use when specifying localised stimuli and
    parameter variations. For efficiently using the whole mesh as sources, such
    as is required to represent local connectivity within the cortex, see the 
    local_gdist_matrix() function.
    
    Basic usage then looks like::
        >>> import numpy
        >>> temp = numpy.loadtxt("flat_triangular_mesh.txt", skiprows=1)
        >>> vertices = temp[0:121].astype(numpy.float64)
        >>> triangles = temp[121:321].astype(numpy.int32)
        >>> src = numpy.array([1], dtype=numpy.int32)
        >>> trg = numpy.array([2], dtype=numpy.int32)
        >>> import gdist
        >>> gdist.compute_gdist(vertices, triangles, source_indices = src, target_indices = trg)
         array([ 0.2])
    
    """
    # Define C++ vectors to contain the mesh surface components.
    cdef vector[double] points
    cdef vector[unsigned] faces
    
    # Map numpy array of mesh "vertices" to C++ vector of mesh "points" 
    cdef numpy.float64_t coord
    for coord in vertices.flatten():
        points.push_back(coord)
    
    # Map numpy array of mesh "triangles" to C++ vector of mesh "faces" 
    cdef numpy.int32_t indx
    for indx in triangles.flatten():
        faces.push_back(indx)

    # Create a mesh object
    cdef Mesh amesh
    amesh.initialize_mesh_data(points, faces)
    
    # Define and create object for algorithm on that mesh
    cdef GeodesicAlgorithmBase *algorithm
    if not approximate:
        algorithm = <GeodesicAlgorithmBase*> new GeodesicAlgorithmExact(&amesh)
    else:
        algorithm = <GeodesicAlgorithmBase*> new GeodesicAlgorithmSubdivision(&amesh, subdivision_level)

    # Define a C++ vector for the source vertices
    cdef vector[SurfacePoint] all_sources
    
    # Define a C++ vector for the target vertices
    cdef vector[SurfacePoint] stop_points
    
    # Define a NumPy array to hold the results
    cdef numpy.ndarray[numpy.float64_t, ndim=1] distances

    cdef Py_ssize_t  k
    
    if source_indices is None: #Default to 0
        source_indices = numpy.arange(0, dtype=numpy.int32)
    # Map the NumPy sources of targets to a C++ vector
    for k in source_indices:
        all_sources.push_back(SurfacePoint(&amesh.vertices()[k]))

    if target_indices is None:
        #Propagte purely on max_distance
        algorithm.propagate(all_sources, max_distance, NULL)
        #Make all vertices targets, NOTE: this is inefficient in the best_source 
        #                                 step below, but not sure how to avoid.
        target_indices = numpy.arange(vertices.shape[0], dtype=numpy.int32)
        # Map the NumPy array of targets to a C++ vector
        for k in target_indices:
            stop_points.push_back(SurfacePoint(&amesh.vertices()[k]))
    else:
        # Map the NumPy array of targets to a C++ vector
        for k in target_indices:
            stop_points.push_back(SurfacePoint(&amesh.vertices()[k]))
        #Propogate to the specified targets
        algorithm.propagate(all_sources, max_distance, &stop_points)
    
    #
    distances = numpy.zeros((len(target_indices), ), dtype=numpy.float64)
    for k in range(stop_points.size()):
        algorithm.best_source(stop_points[k], distances[k])
    
    distances[distances==GEODESIC_INF] = numpy.inf

    del algorithm
    return distances

def local_gdist_matrix(numpy.ndarray[numpy.float64_t, ndim=2] vertices,
                       numpy.ndarray[numpy.int32_t, ndim=2] triangles,
                       double max_distance = GEODESIC_INF):
    """
    This is the wrapper function for computing geodesic distance from every 
    vertex on the surface to all those within a distance ``max_distance`` of 
    them. The function accepts three arguments:
        ``vertices``: defines x,y,z coordinates of the mesh's vertices
        ``triangles``: defines faces of the mesh as index triplets into vertices.
        ``max_distance``: 
    and returns a scipy.sparse.csc_matrix((N, N), dtype=numpy.float64), where N
    is the number of vertices, specifying the shortest distance from all 
    vertices to all the vertices within max_distance.
    
    Basic usage then looks like::
        >>> import numpy
        >>> temp = numpy.loadtxt("flat_triangular_mesh.txt", skiprows=1)
        >>> import gdist
        >>> vertices = temp[0:121].astype(numpy.float64)
        >>> triangles = temp[121:321].astype(numpy.int32)
        >>> gdist.local_gdist_matrix(vertices, triangles, max_distance=1.0)
         <121x121 sparse matrix of type '<type 'numpy.float64'>'
             with 6232 stored elements in Compressed Sparse Column format>

    Runtime and guesstimated memory usage as a function of max_distance for the
    reg_13 cortical surface mesh, ie containing 2**13 vertices per hemisphere.
    :: 
        [[10, 20, 30, 40,  50,  60,  70,  80,  90, 100], # mm
         [19, 28, 49, 81, 125, 181, 248, 331, 422, 522], # s
         [ 3, 13, 30, 56,  89, 129, 177, 232, 292, 358]] # MB]
         
    where memory is a min-guestimate given by: mem_req = nnz * 8 / 1024 / 1024
    
    NOTE: The best_source loop could be sped up considerably by restricting 
          targets to those vertices within max_distance of the source, however,
          this will first require the efficient extraction of this information 
          from the propgate step...
    """
    
    # Define C++ vectors to contain the mesh surface components.
    cdef vector[double] points
    cdef vector[unsigned] faces
    
    # Map numpy array of mesh "vertices" to C++ vector of mesh "points" 
    cdef numpy.float64_t coord
    for coord in vertices.flatten():
        points.push_back(coord)
    
    # Map numpy array of mesh "triangles" to C++ vector of mesh "faces" 
    cdef numpy.int32_t indx
    for indx in triangles.flatten():
        faces.push_back(indx)

    # Create a mesh object
    cdef Mesh amesh
    amesh.initialize_mesh_data(points, faces)
    
    # Define and create object for exact algorithm on that mesh:
    cdef GeodesicAlgorithmExact *algorithm = new GeodesicAlgorithmExact(&amesh)
    
    cdef vector[SurfacePoint] source, targets
    cdef Py_ssize_t N = vertices.shape[0]
    cdef Py_ssize_t k
    cdef Py_ssize_t kk
    cdef numpy.float64_t distance
    
    # Add all vertices as targets
    for k in range(N):
        targets.push_back(SurfacePoint(&amesh.vertices()[k]))
    
    #
    rows = []
    columns = []
    data = []
    for k in range(N):
        source.push_back(SurfacePoint(&amesh.vertices()[k]))
        algorithm.propagate(source, max_distance, NULL)
        source.pop_back()
        
        for kk in range(N): #TODO: Reduce to vertices reached during propagate.
            algorithm.best_source(targets[kk], distance)
            
            if (distance is not GEODESIC_INF) and (distance is not 0):
                rows.append(k)
                columns.append(kk)
                data.append(distance)
    
    return scipy.sparse.csc_matrix((data, (rows, columns)), shape=(N, N))
