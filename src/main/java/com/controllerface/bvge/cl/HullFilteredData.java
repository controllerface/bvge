package com.controllerface.bvge.cl;

import org.jocl.cl_mem;

public record HullFilteredData(cl_mem hulls_out, int hull_count) {

}
