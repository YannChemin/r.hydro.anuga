/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Shallow-water flood simulation with ANUGA's discontinuous
 *               elevation (DE) finite-volume scheme in OpenCL, on a
 *               multi-resolution triangular mesh built from one or more
 *               DEMs, with rainfall/ERA5 forcing and Green-Ampt
 *               infiltration. See PLAN.md for the design.
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/raster.h>

#include "evolve.h"
#include "kernels_ocl.h"
#include "mesh.h"
#include "ocl_backend.h"
#include "output.h"
#include "preflight.h"
#include "quadtree.h"
#include "sampler.h"
#include "setup.h"
#include "state.h"

struct options {
    /* Terrain and mesh. */
    struct Option *elevation, *domain, *dem_offset, *dem_bias_tolerance,
        *blend_width, *res_min, *res_max, *fringe, *refine, *refine_res,
        *coarsen, *relief_tolerance, *max_triangles;
    /* Surface. */
    struct Option *manning, *manning_value, *landcover, *manning_rules,
        *buildings, *building_height;
    /* Initial conditions. */
    struct Option *initial_depth, *initial_stage, *initial_xmom, *initial_ymom;
    /* Infiltration. */
    struct Option *infiltration, *soil_texture, *soil_table, *soil_horizon, *ks,
        *suction, *porosity, *initial_saturation, *soil_depth, *impervious,
        *infil_substep;
    /* Forcing. */
    struct Option *rain, *rain_strds, *rain_value, *rain_hyetograph,
        *rain_units, *evap, *evap_strds, *evap_units, *wind_u_strds,
        *wind_v_strds, *pressure_strds, *pressure_units, *inflow,
        *inflow_series;
    /* Boundaries. */
    struct Option *boundary, *stage_series;
    /* Time and solver. */
    struct Option *start, *end, *duration, *output_step, *max_timestep,
        *min_timestep, *cfl, *algorithm, *friction_method;
    /* Outputs. */
    struct Option *output, *outputs, *min_depth, *max_depth, *max_speed,
        *max_stage, *max_hazard, *arrival_time, *arrival_depth,
        *inundation_duration, *final_prefix, *mesh_level, *mesh_output,
        *state_output, *gauges, *gauge_output, *gauge_step, *massbalance;
    /* Backend and reporting. */
    struct Option *device, *nprocs, *format;
};

struct flags {
    struct Flag *dry_run, *keep_order, *fringe_inside, *detail, *null_dry,
        *dcell, *mass_report, *ks_mm_per_hour;
};

static struct Option *opt_raster_in(const char *key, const char *label,
                                    const char *section)
{
    struct Option *o = G_define_standard_option(G_OPT_R_INPUT);

    o->key = key;
    o->required = NO;
    o->label = label;
    o->guisection = section;

    return o;
}

static struct Option *opt_raster_out(const char *key, const char *label,
                                     const char *section)
{
    struct Option *o = G_define_standard_option(G_OPT_R_OUTPUT);

    o->key = key;
    o->required = NO;
    o->label = label;
    o->guisection = section;

    return o;
}

static struct Option *opt_strds_in(const char *key, const char *label,
                                   const char *section)
{
    struct Option *o = G_define_standard_option(G_OPT_STRDS_INPUT);

    o->key = key;
    o->required = NO;
    o->label = label;
    o->guisection = section;

    return o;
}

static struct Option *opt_value(const char *key, int type, const char *answer,
                                const char *label, const char *section)
{
    struct Option *o = G_define_option();

    o->key = key;
    o->type = type;
    o->required = NO;
    o->answer = (char *)answer;
    o->label = label;
    o->guisection = section;

    return o;
}

static struct Option *opt_choice(const char *key, const char *options,
                                 const char *answer, const char *label,
                                 const char *section)
{
    struct Option *o = opt_value(key, TYPE_STRING, answer, label, section);

    o->options = options;

    return o;
}

static struct Flag *flag(char key, const char *description, const char *section)
{
    struct Flag *f = G_define_flag();

    f->key = key;
    f->description = description;
    f->guisection = section;

    return f;
}

static void define_options(struct options *opt, struct flags *flg)
{
    const char *terrain = _("Terrain"), *surface = _("Surface"),
               *initial = _("Initial"), *infil = _("Infiltration"),
               *forcing = _("Forcing"), *bounds = _("Boundaries"),
               *timing = _("Time"), *out = _("Output"), *backend = _("Backend");

    /* Terrain and mesh (PLAN.md section 4). */
    opt->elevation = G_define_standard_option(G_OPT_R_ELEVS);
    opt->elevation->description =
        _("One or more DEMs; the mesh uses the finest DEM available at each "
          "location");
    opt->elevation->guisection = terrain;

    opt->domain = opt_raster_in(
        "domain", _("Raster map defining the active domain (non-NULL cells)"),
        terrain);
    opt->dem_offset = opt_value(
        "dem_offset", TYPE_DOUBLE, NULL,
        _("Explicit vertical offset per DEM (m), added to its values"),
        terrain);
    opt->dem_offset->multiple = YES;
    opt->dem_bias_tolerance =
        opt_value("dem_bias_tolerance", TYPE_DOUBLE, "0.5",
                  _("Maximum allowed median vertical difference between "
                    "overlapping DEMs (m)"),
                  terrain);
    opt->blend_width = opt_value(
        "blend_width", TYPE_DOUBLE, "2",
        _("Width of the seam blending zone, in coarse DEM cells"), terrain);
    opt->res_min = opt_value(
        "res_min", TYPE_DOUBLE, NULL,
        _("Finest mesh cell size (default: finest DEM resolution)"), terrain);
    opt->res_max =
        opt_value("res_max", TYPE_DOUBLE, NULL,
                  _("Coarsest mesh cell size, snapped to res_min*2^L (default: "
                    "coarsest DEM resolution)"),
                  terrain);
    opt->fringe = opt_value(
        "fringe", TYPE_INTEGER, "4",
        _("Minimum number of cells of each level between level changes"),
        terrain);
    opt->refine = opt_raster_in(
        "refine", _("Raster map of areas to refine to refine_res"), terrain);
    opt->refine_res =
        opt_value("refine_res", TYPE_DOUBLE, NULL,
                  _("Target cell size inside refine areas"), terrain);
    opt->coarsen =
        opt_raster_in("coarsen",
                      _("Raster map of areas where the mesh may stay "
                        "coarser than the DEM (value = cell size)"),
                      terrain);
    opt->relief_tolerance = opt_value(
        "relief_tolerance", TYPE_DOUBLE, NULL,
        _("Coarsen cells whose sub-cell relief is below this value (m)"),
        terrain);
    opt->max_triangles = opt_value(
        "max_triangles", TYPE_DOUBLE, NULL,
        _("Fail if the mesh would exceed this number of triangles"), terrain);

    /* Surface (PLAN.md section 7.1). */
    opt->manning =
        opt_raster_in("manning", _("Raster map of Manning's n"), surface);
    opt->manning_value = opt_value("manning_value", TYPE_DOUBLE, "0.03",
                                   _("Constant Manning's n"), surface);
    opt->landcover = opt_raster_in(
        "landcover", _("Land cover raster map (with manning_rules)"), surface);
    opt->manning_rules = G_define_standard_option(G_OPT_F_INPUT);
    opt->manning_rules->key = "manning_rules";
    opt->manning_rules->required = NO;
    opt->manning_rules->label =
        _("File with 'class n' lines mapping land cover to Manning's n");
    opt->manning_rules->guisection = surface;
    opt->buildings = opt_raster_in(
        "buildings", _("Raster map of building footprints"), surface);
    opt->building_height =
        opt_value("building_height", TYPE_DOUBLE, "10",
                  _("Height added to elevation under buildings (m)"), surface);

    /* Initial conditions. */
    opt->initial_depth = opt_raster_in(
        "initial_depth", _("Raster map of initial water depth (m)"), initial);
    opt->initial_stage = opt_raster_in(
        "initial_stage", _("Raster map of initial water surface (m)"), initial);
    opt->initial_xmom = opt_raster_in(
        "initial_xmom", _("Raster map of initial x momentum (m2/s)"), initial);
    opt->initial_ymom = opt_raster_in(
        "initial_ymom", _("Raster map of initial y momentum (m2/s)"), initial);

    /* Infiltration (PLAN.md section 6). */
    opt->infiltration = opt_choice("infiltration", "none,ga,gar", "none",
                                   _("Infiltration model"), infil);
    opt->infiltration->descriptions =
        _("none;No infiltration;"
          "ga;Green-Ampt with ponded head, Rawls et al. (1983) parameters;"
          "gar;Green-Ampt with redistribution, Ogden and Saghafian (1997)");
    opt->soil_texture =
        opt_raster_in("soil_texture",
                      _("Raster map of USDA soil texture classes 1-11"), infil);
    opt->soil_table = G_define_standard_option(G_OPT_F_INPUT);
    opt->soil_table->key = "soil_table";
    opt->soil_table->required = NO;
    opt->soil_table->label = _("CSV soil parameter table overriding the "
                               "built-in one");
    opt->soil_table->guisection = infil;
    opt->soil_horizon = opt_choice(
        "soil_horizon", "all,A,B,C", "all",
        _("Soil horizon rows of Rawls et al. (1983) Table 2"), infil);
    opt->ks = opt_raster_in(
        "ks", _("Raster map of saturated hydraulic conductivity (m/s)"), infil);
    opt->suction = opt_raster_in(
        "suction", _("Raster map of wetting front suction head (m)"), infil);
    opt->porosity =
        opt_raster_in("porosity", _("Raster map of effective porosity"), infil);
    opt->initial_saturation = opt_value(
        "initial_saturation", TYPE_STRING, "0",
        _("Initial effective saturation 0-1 (value or raster)"), infil);
    opt->soil_depth = opt_raster_in(
        "soil_depth", _("Raster map of depth to an impervious layer (m)"),
        infil);
    opt->impervious = opt_raster_in(
        "impervious", _("Raster map of impervious fraction 0-1"), infil);
    opt->infil_substep =
        opt_value("infil_substep", TYPE_DOUBLE, "30",
                  _("Soil water update interval for GAR (seconds)"), infil);

    /* Forcing (PLAN.md section 7.2). */
    opt->rain =
        opt_raster_in("rain", _("Raster map of rainfall rate"), forcing);
    opt->rain_strds = opt_strds_in(
        "rain_strds", _("Space-time raster dataset of rainfall"), forcing);
    opt->rain_value =
        opt_value("rain_value", TYPE_DOUBLE, NULL,
                  _("Constant, spatially uniform rainfall rate"), forcing);
    opt->rain_hyetograph = G_define_standard_option(G_OPT_F_INPUT);
    opt->rain_hyetograph->key = "rain_hyetograph";
    opt->rain_hyetograph->required = NO;
    opt->rain_hyetograph->label =
        _("CSV file of 'seconds,rate' for spatially uniform rainfall");
    opt->rain_hyetograph->guisection = forcing;
    opt->rain_units = opt_choice("rain_units", "mm/h,mm/d,m/s", NULL,
                                 _("Units of the rainfall input"), forcing);
    opt->evap = opt_raster_in(
        "evap", _("Raster map of (potential) evaporation rate"), forcing);
    opt->evap_strds = opt_strds_in(
        "evap_strds", _("Space-time raster dataset of evaporation"), forcing);
    opt->evap_units = opt_choice("evap_units", "mm/h,mm/d,m/s", NULL,
                                 _("Units of the evaporation input"), forcing);
    opt->wind_u_strds = opt_strds_in(
        "wind_u_strds", _("Space-time raster dataset of 10 m U wind (m/s)"),
        forcing);
    opt->wind_v_strds = opt_strds_in(
        "wind_v_strds", _("Space-time raster dataset of 10 m V wind (m/s)"),
        forcing);
    opt->pressure_strds = opt_strds_in(
        "pressure_strds", _("Space-time raster dataset of surface pressure"),
        forcing);
    opt->pressure_units = opt_choice("pressure_units", "Pa,kPa", NULL,
                                     _("Units of the pressure input"), forcing);
    opt->inflow = G_define_standard_option(G_OPT_V_INPUT);
    opt->inflow->key = "inflow";
    opt->inflow->required = NO;
    opt->inflow->label = _("Vector points of inflow locations");
    opt->inflow->guisection = forcing;
    opt->inflow_series = G_define_standard_option(G_OPT_F_INPUT);
    opt->inflow_series->key = "inflow_series";
    opt->inflow_series->required = NO;
    opt->inflow_series->label =
        _("CSV file of 'seconds,cat,discharge' inflow hydrographs (m3/s)");
    opt->inflow_series->guisection = forcing;

    /* Boundaries (PLAN.md section 5.2). */
    opt->boundary =
        opt_value("boundary", TYPE_STRING,
                  "north:reflective,south:reflective,"
                  "east:reflective,west:reflective,"
                  "null:reflective",
                  _("Boundary condition per side, side:type[:value]"), bounds);
    opt->boundary->multiple = YES;
    opt->boundary->key_desc = "side:type[:value]";
    opt->stage_series = G_define_standard_option(G_OPT_F_INPUT);
    opt->stage_series->key = "stage_series";
    opt->stage_series->required = NO;
    opt->stage_series->label =
        _("CSV file of 'seconds,stage' for stage/flather boundaries");
    opt->stage_series->guisection = bounds;

    /* Time and solver (PLAN.md section 7.3). */
    opt->start = opt_value("start", TYPE_STRING, NULL,
                           _("Start date and time (ISO 8601; default: first "
                             "forcing map)"),
                           timing);
    opt->end = opt_value("end", TYPE_STRING, NULL,
                         _("End date and time (ISO 8601)"), timing);
    opt->duration = opt_value("duration", TYPE_DOUBLE, NULL,
                              _("Simulated duration (seconds)"), timing);
    opt->output_step =
        opt_value("output_step", TYPE_DOUBLE, "3600",
                  _("Interval between outputs (seconds)"), timing);
    opt->max_timestep = opt_value("max_timestep", TYPE_DOUBLE, NULL,
                                  _("Maximum time step (seconds)"), timing);
    opt->min_timestep = opt_value(
        "min_timestep", TYPE_DOUBLE, "1e-6",
        _("Minimum time step before a stall is reported (s)"), timing);
    opt->cfl = opt_value(
        "cfl", TYPE_DOUBLE, NULL,
        _("CFL number (default: DE0 0.9, DE1 0.5, DE2 1.0)"), timing);
    opt->algorithm = opt_choice("algorithm", "DE0,DE1,DE2", "DE1",
                                _("ANUGA flow algorithm"), timing);
    opt->algorithm->descriptions = _("DE0;Euler, CFL 0.9, beta 0.5;"
                                     "DE1;Runge-Kutta 2, CFL 1.0, beta 1.0;"
                                     "DE2;Runge-Kutta 3");
    opt->friction_method =
        opt_choice("friction_method", "flat,sloped", "flat",
                   _("Manning friction formulation"), timing);

    /* Outputs (PLAN.md section 8). */
    opt->output = opt_value("output", TYPE_STRING, NULL,
                            _("Base name for output space-time raster "
                              "datasets"),
                            out);
    opt->output->gisprompt = "new,strds,strds";
    opt->outputs =
        opt_choice("outputs",
                   "depth,stage,xvelocity,yvelocity,speed,direction,discharge,"
                   "xmomentum,ymomentum,froude,hazard,infiltration",
                   "depth", _("Quantities written as time series"), out);
    opt->outputs->multiple = YES;
    opt->min_depth =
        opt_value("min_depth", TYPE_DOUBLE, "0.001",
                  _("Depths below this are written as dry (m)"), out);
    opt->max_depth = opt_raster_out("max_depth", _("Maximum depth"), out);
    opt->max_speed = opt_raster_out("max_speed", _("Maximum speed"), out);
    opt->max_stage = opt_raster_out("max_stage", _("Maximum stage"), out);
    opt->max_hazard =
        opt_raster_out("max_hazard", _("Maximum hazard rating"), out);
    opt->arrival_time = opt_raster_out(
        "arrival_time", _("Time water first exceeds arrival_depth (s)"), out);
    opt->arrival_depth =
        opt_value("arrival_depth", TYPE_DOUBLE, "0.05",
                  _("Depth threshold for arrival and duration (m)"), out);
    opt->inundation_duration =
        opt_raster_out("inundation_duration",
                       _("Total time with depth above arrival_depth (s)"), out);
    opt->final_prefix = opt_value(
        "final_prefix", TYPE_STRING, NULL,
        _("Prefix for final state rasters (stage, xmom, ymom) for hot start"),
        out);
    opt->mesh_level = opt_raster_out(
        "mesh_level", _("Quadtree level used at each cell"), out);
    opt->mesh_output = G_define_standard_option(G_OPT_F_OUTPUT);
    opt->mesh_output->key = "mesh_output";
    opt->mesh_output->required = NO;
    opt->mesh_output->label =
        _("Directory to export the mesh and scaled elevation to");
    opt->mesh_output->guisection = out;
    opt->state_output = G_define_standard_option(G_OPT_F_OUTPUT);
    opt->state_output->key = "state_output";
    opt->state_output->required = NO;
    opt->state_output->label =
        _("Directory to export the initial and final solver state to");
    opt->state_output->description =
        _("Raw centroid arrays and the time step log, for validation");
    opt->state_output->guisection = out;
    opt->gauges = G_define_standard_option(G_OPT_V_INPUT);
    opt->gauges->key = "gauges";
    opt->gauges->required = NO;
    opt->gauges->label = _("Vector points where time series are recorded");
    opt->gauges->guisection = out;
    opt->gauge_output = G_define_standard_option(G_OPT_F_OUTPUT);
    opt->gauge_output->key = "gauge_output";
    opt->gauge_output->required = NO;
    opt->gauge_output->label = _("CSV file for gauge time series");
    opt->gauge_output->guisection = out;
    opt->gauge_step = opt_value("gauge_step", TYPE_DOUBLE, "60",
                                _("Interval between gauge records (s)"), out);
    opt->massbalance = G_define_standard_option(G_OPT_F_OUTPUT);
    opt->massbalance->key = "massbalance";
    opt->massbalance->required = NO;
    opt->massbalance->label = _("CSV file for the mass balance report");
    opt->massbalance->guisection = out;

    /* Backend. */
    opt->device = opt_choice("device", "auto,gpu,cpu,omp", "auto",
                             _("Compute backend"), backend);
    opt->device->descriptions = _("auto;GPU, then CPU OpenCL, then OpenMP;"
                                  "gpu;OpenCL GPU with double precision;"
                                  "cpu;OpenCL CPU device (e.g. PoCL);"
                                  "omp;OpenMP, no OpenCL");
    opt->nprocs = G_define_standard_option(G_OPT_M_NPROCS);
    opt->nprocs->guisection = backend;
    opt->format = G_define_standard_option(G_OPT_F_FORMAT);
    opt->format->options = "plain,shell";
    opt->format->descriptions =
        _("plain;Human readable text;shell;shell script style key=value");
    opt->format->guisection = backend;

    flg->dry_run = flag('p',
                        _("Print device, mesh levels and memory estimate, "
                          "then exit"),
                        backend);
    flg->keep_order =
        flag('o',
             _("Use the elevation maps in the given priority order "
               "instead of finest first"),
             terrain);
    flg->fringe_inside =
        flag('i', _("Place the resolution fringe inside fine DEM footprints"),
             terrain);
    flg->detail = flag(
        'f', _("Also write fine-resolution outputs over fine DEM footprints"),
        out);
    flg->null_dry = flag('n', _("Write dry cells as NULL instead of 0"), out);
    flg->dcell = flag('d', _("Write DCELL instead of FCELL outputs"), out);
    flg->mass_report =
        flag('m', _("Print the mass balance error at the end"), out);
    flg->ks_mm_per_hour =
        flag('k', _("Values of ks= are in mm/h instead of m/s"), infil);

    G_option_exclusive(opt->initial_depth, opt->initial_stage, NULL);
    G_option_exclusive(opt->rain, opt->rain_strds, opt->rain_value,
                       opt->rain_hyetograph, NULL);
    G_option_exclusive(opt->evap, opt->evap_strds, NULL);
    G_option_exclusive(opt->manning, opt->landcover, NULL);
    G_option_requires(opt->landcover, opt->manning_rules, NULL);
    G_option_requires(opt->refine, opt->refine_res, NULL);
    G_option_requires(opt->wind_u_strds, opt->wind_v_strds, NULL);
    G_option_requires(opt->wind_v_strds, opt->wind_u_strds, NULL);
    G_option_requires(opt->gauges, opt->gauge_output, NULL);
    G_option_exclusive(opt->end, opt->duration, NULL);
}

static double parse_positive(const struct Option *o)
{
    double v;

    if (!o->answer)
        return 0.0;
    v = atof(o->answer);
    if (!(v > 0.0))
        G_fatal_error(_("Option %s= must be positive, got '%s'"), o->key,
                      o->answer);

    return v;
}

/* Checks that do not need any data. Units are never guessed. */
static void validate(const struct options *opt)
{
    if ((opt->rain->answer || opt->rain_strds->answer ||
         opt->rain_value->answer || opt->rain_hyetograph->answer) &&
        !opt->rain_units->answer)
        G_fatal_error(_("Rainfall input given without rain_units= "
                        "(choose mm/h, mm/d or m/s; t.in.era5 produces mm/d, "
                        "or mm/h with its -h flag)"));
    if ((opt->evap->answer || opt->evap_strds->answer) &&
        !opt->evap_units->answer)
        G_fatal_error(_("Evaporation input given without evap_units= "
                        "(choose mm/h, mm/d or m/s)"));
    if (opt->pressure_strds->answer && !opt->pressure_units->answer)
        G_fatal_error(_("pressure_strds= given without pressure_units= "
                        "(t.in.era5 surface_pressure is in kPa)"));
    if (strcmp(opt->infiltration->answer, "none") != 0 &&
        !opt->soil_texture->answer &&
        !(opt->ks->answer && opt->suction->answer && opt->porosity->answer))
        G_fatal_error(_("infiltration=%s needs soil_texture= or all of ks=, "
                        "suction= and porosity="),
                      opt->infiltration->answer);
    if (atoi(opt->fringe->answer) < 1)
        G_fatal_error(_("fringe= must be at least 1"));
    parse_positive(opt->res_min);
    parse_positive(opt->res_max);
    parse_positive(opt->output_step);
}

static int count_max_outputs(const struct options *opt)
{
    return (opt->max_depth->answer != NULL) + (opt->max_speed->answer != NULL) +
           (opt->max_stage->answer != NULL) +
           (opt->max_hazard->answer != NULL) +
           (opt->arrival_time->answer != NULL) +
           (opt->inundation_duration->answer != NULL);
}

static void print_backend(const struct ocl_backend *backend, const char *format)
{
    if (format && strcmp(format, "shell") == 0) {
        fprintf(stdout, "backend=%s\n", ocl_backend_tier_name(backend));
        if (backend->tier == DEV_OMP)
            return;
        fprintf(stdout,
                "device=%s\nplatform=%s\ndevice_version=%s\n"
                "device_global_mem=%llu\ndevice_max_alloc=%llu\n"
                "device_compute_units=%u\n",
                backend->device_name, backend->platform_name,
                backend->device_version,
                (unsigned long long)backend->global_mem_size,
                (unsigned long long)backend->max_alloc_size,
                backend->compute_units);
        return;
    }
    if (backend->tier == DEV_OMP) {
        fprintf(stdout, "Backend: OpenMP (no OpenCL device)\n");
        return;
    }
    fprintf(stdout,
            "Backend: OpenCL %s '%s' on '%s' (%s), %u compute units, "
            "%.2f GiB, max buffer %.2f GiB\n",
            ocl_backend_tier_name(backend), backend->device_name,
            backend->platform_name, backend->device_version,
            backend->compute_units, backend->global_mem_size / 1073741824.0,
            backend->max_alloc_size / 1073741824.0);
}

/* Compare the estimate with the device and the triangle budget. */
static void check_capacity(const struct preflight *pf,
                           const struct ocl_backend *backend,
                           const struct options *opt)
{
    if (opt->max_triangles->answer &&
        (double)pf->triangles > atof(opt->max_triangles->answer))
        G_fatal_error(_("Estimated %lld triangles exceed max_triangles=%s; "
                        "increase res_min=, use coarsen=, or reduce the "
                        "region"),
                      pf->triangles, opt->max_triangles->answer);
    if (backend->tier == DEV_OMP)
        return;
    if (pf->device_bytes > 0.9 * (double)backend->global_mem_size)
        G_warning(_("Estimated device memory %.2f GiB exceeds 90%% of the "
                    "device's %.2f GiB; increase res_min=, use coarsen= or "
                    "reduce the region"),
                  pf->device_bytes / 1073741824.0,
                  backend->global_mem_size / 1073741824.0);
    if (pf->largest_buffer_bytes > (double)backend->max_alloc_size)
        G_warning(_("Largest buffer %.2f GiB exceeds the device's maximum "
                    "allocation of %.2f GiB; buffers will need to be split"),
                  pf->largest_buffer_bytes / 1073741824.0,
                  backend->max_alloc_size / 1073741824.0);
}

/* Leaf activity for the uniform quadtree: the leaf must contain valid
 * DEM cells and, if a domain map is given, its centre must be inside. */
struct activity {
    const struct raster_grid *dem;
    const struct raster_grid *domain;
};

static int leaf_is_active(const struct quadtree *qt, const struct leaf *lf,
                          void *data)
{
    const struct activity *act = data;
    double size = quadtree_leaf_size(qt, lf), x, y;
    long count;

    quadtree_leaf_origin(qt, lf, &x, &y);
    x += qt->west;
    y += qt->south;
    if (act->domain &&
        isnan(raster_grid_nearest(act->domain, x + 0.5 * size, y + 0.5 * size)))
        return 0;
    raster_grid_box_mean(act->dem, x, y, x + size, y + size, &count);

    return count > 0;
}

/* Run the solver on the mesh and export the state (phase 2: OpenMP tier,
 * no raster time series yet). */
static void run_solver(const struct options *opt, const struct mesh *mesh,
                       const struct ocl_backend *backend)
{
    struct solver_config cfg;
    struct sw_state state;
    struct state_snapshot initial, final;
    struct evolve_log log = {0};
    const struct solver_ops *ops =
        backend->tier == DEV_OMP ? &omp_ops : &ocl_ops;
    double duration, yieldstep;

    if (!opt->duration->answer)
        G_fatal_error(_("duration= is required to run the solver"));
    duration = atof(opt->duration->answer);
    yieldstep = atof(opt->output_step->answer);

    setup_config(&cfg, opt->algorithm->answer, opt->cfl->answer,
                 opt->friction_method->answer);
    state_init(&state, mesh, &cfg);
    setup_boundaries(&state, mesh, opt->boundary->answers);
    setup_initial(&state, mesh, opt->initial_depth->answer,
                  opt->initial_stage->answer);
    setup_friction(&state, mesh, opt->manning->answer,
                   atof(opt->manning_value->answer));

    if (ops == &ocl_ops)
        ocl_solver_init(&state, backend);
    snapshot_take(&initial, &state, ops);
    G_message(_("Running %s for %g s (%s tier)..."), opt->algorithm->answer,
              duration, ops->name);
    evolve_run(&state, ops, duration, yieldstep, NULL, NULL, &log);
    snapshot_take(&final, &state, ops);
    G_message(_("%ld time steps; water volume %.6g -> %.6g m3 (signed %.6g "
                "-> %.6g m3), boundary inflow %.6g m3, added by "
                "negative-depth clamping %.6g m3"),
              log.n_steps, initial.volume, final.volume, initial.signed_volume,
              final.signed_volume, log.boundary_mass, log.protect_mass);

    /* Clamping negative depths creates water; say so when it matters. */
    if (log.protect_mass > 0.01 * fmax(initial.signed_volume, 0.0) &&
        log.protect_mass > 0.0)
        G_warning(_("Clamping negative depths added %.6g m3 of water (%.2f%% "
                    "of the initial volume). Consider a lower cfl= or "
                    "algorithm=DE0"),
                  log.protect_mass,
                  initial.signed_volume > 0.0
                      ? 100.0 * log.protect_mass / initial.signed_volume
                      : 100.0);

    if (opt->state_output->answer)
        state_export(opt->state_output->answer, &state, &initial, &final, &log,
                     duration, yieldstep);

    if (state.device)
        ocl_solver_free(&state);
    snapshot_free(&initial);
    snapshot_free(&final);
    evolve_log_free(&log);
    state_free(&state);
}

/* Build the mesh, check the device dequantisation, write the requested
 * mesh outputs, and run the solver if a state output is requested. */
static void build_and_run(const struct options *opt, const struct preflight *pf,
                          const struct ocl_backend *backend)
{
    struct raster_grid dem, domain;
    struct activity act;
    struct quadtree qt;
    struct mesh mesh;
    double *z;
    long k, mismatches;

    if (pf->n_dems > 1)
        G_fatal_error(_("Meshes from several DEMs are not implemented yet "
                        "(phase 5 of PLAN.md); give a single elevation map"));
    if (pf->n_levels > 1)
        G_fatal_error(_("Multi-level meshes are not implemented yet (phase 5 "
                        "of PLAN.md); res_min and res_max must be equal"));

    G_message(_("Reading elevation map <%s>..."), pf->dems[0].name);
    raster_grid_load(&dem, pf->dems[0].name);
    act.dem = &dem;
    act.domain = NULL;
    if (opt->domain->answer) {
        raster_grid_load(&domain, opt->domain->answer);
        act.domain = &domain;
    }

    G_message(_("Selecting active cells..."));
    quadtree_build_uniform(&qt, pf->res_max, leaf_is_active, &act);
    mesh_build(&mesh, &qt, &dem);
    G_message(_("Mesh: %ld triangles, %ld nodes, %ld boundary edges"),
              mesh.n_tri, mesh.n_nodes, mesh.n_boundary);
    G_message(_("Bed datum %.4f m, maximum quantization error %.3g m"),
              (double)mesh.z0 * BED_INV_SCALE, mesh.max_quantization_error);

    /* The device must reproduce the host's dequantised bed bit for bit. */
    if (backend->tier != DEV_OMP) {
        z = G_malloc(mesh.n_tri * sizeof(double));
        for (k = 0; k < mesh.n_tri; k++)
            z[k] = bed_dequantize(mesh.zq[k], mesh.z0);
        mismatches =
            ocl_check_dequantize(backend, mesh.zq, mesh.n_tri, mesh.z0, z);
        G_free(z);
        if (mismatches)
            G_fatal_error(_("OpenCL device '%s' dequantises %ld of %ld bed "
                            "values differently from the host"),
                          backend->device_name, mismatches, mesh.n_tri);
        G_verbose_message(_("OpenCL dequantised bed matches the host bit for "
                            "bit"));
    }

    if (opt->mesh_output->answer)
        mesh_export(&mesh, &qt, opt->mesh_output->answer);
    if (opt->mesh_level->answer)
        output_mesh_level(&qt, opt->mesh_level->answer);
    if (opt->state_output->answer)
        run_solver(opt, &mesh, backend);

    mesh_free(&mesh);
    quadtree_free(&qt);
    raster_grid_free(&dem);
    if (act.domain)
        raster_grid_free(&domain);
}

int main(int argc, char *argv[])
{
    struct GModule *module;
    struct options opt;
    struct flags flg;
    struct ocl_backend backend;
    struct preflight pf;
    struct memory_options mopt;

    G_gisinit(argv[0]);

    module = G_define_module();
    G_add_keyword(_("raster"));
    G_add_keyword(_("hydrology"));
    G_add_keyword(_("flood"));
    G_add_keyword(_("shallow water"));
    G_add_keyword(_("ANUGA"));
    G_add_keyword(_("OpenCL"));
    G_add_keyword(_("GPU"));
    module->description =
        _("Simulates overland flow and flooding with ANUGA's shallow-water "
          "solver on a multi-resolution mesh, accelerated with OpenCL.");

    define_options(&opt, &flg);

    if (G_parser(argc, argv))
        exit(EXIT_FAILURE);

    validate(&opt);

    if (!ocl_backend_init(&backend, opt.device->answer))
        G_fatal_error(_("Unable to initialize the '%s' compute backend"),
                      opt.device->answer);

    memset(&pf, 0, sizeof(pf));
    preflight_describe_dems(&pf, opt.elevation->answers,
                            flg.keep_order->answer);
    preflight_levels(&pf, opt.res_min->answer ? atof(opt.res_min->answer) : 0,
                     opt.res_max->answer ? atof(opt.res_max->answer) : 0);
    mopt.infiltration = strcmp(opt.infiltration->answer, "ga") == 0    ? 1
                        : strcmp(opt.infiltration->answer, "gar") == 0 ? 2
                                                                       : 0;
    mopt.n_max_outputs = count_max_outputs(&opt);
    preflight_estimate(&pf, opt.domain->answer, atoi(opt.fringe->answer),
                       &mopt);

    if (flg.dry_run->answer) {
        print_backend(&backend, opt.format->answer);
        preflight_print(&pf, opt.format->answer);
        check_capacity(&pf, &backend, &opt);
        ocl_backend_free(&backend);
        exit(EXIT_SUCCESS);
    }

    check_capacity(&pf, &backend, &opt);

    if (opt.output->answer)
        G_fatal_error(_("Raster time series outputs (output=) are not "
                        "implemented yet (phase 4 of PLAN.md). Use -p, "
                        "mesh_output=/mesh_level=, or state_output= with "
                        "duration= to run the solver."));
    if (!opt.mesh_output->answer && !opt.mesh_level->answer &&
        !opt.state_output->answer)
        G_fatal_error(_("Nothing to do: give output= (simulation), "
                        "state_output= (solver state), mesh_output= or "
                        "mesh_level= (mesh only), or -p"));

    G_set_omp_num_threads(opt.nprocs);
    build_and_run(&opt, &pf, &backend);
    ocl_backend_free(&backend);

    exit(EXIT_SUCCESS);
}
