/**\file */
#ifndef SLIC_DECLARATIONS_FloatForwardProp_H
#define SLIC_DECLARATIONS_FloatForwardProp_H
#include "MaxSLiCInterface.h"
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define FloatForwardProp_OUT_VEC_SIZE1 (8)
#define FloatForwardProp_OUT_VEC_SIZE2 (1)
#define FloatForwardProp_SIZE_LAYER_2 (10)
#define FloatForwardProp_SIZE_LAYER_0 (784)
#define FloatForwardProp_DYNAMIC_CLOCKS_ENABLED (0)
#define FloatForwardProp_SIZE_LAYER_1 (64)
#define FloatForwardProp_PCIE_ALIGNMENT (16)
#define FloatForwardProp_IN_VEC_SIZE1 (16)
#define FloatForwardProp_IN_VEC_SIZE2 (16)


/*----------------------------------------------------------------------------*/
/*---------------------------- Interface default -----------------------------*/
/*----------------------------------------------------------------------------*/



/**
 * \brief Auxiliary function to evaluate expression for "FOUTPUTLAYER_KERNEL.offset".
 */
int FloatForwardProp_get_FOUTPUTLAYER_KERNEL_offset( void );

/**
 * \brief Auxiliary function to evaluate expression for "FHIDDENLAYER_KERNEL.offset".
 */
int FloatForwardProp_get_FHIDDENLAYER_KERNEL_offset( void );


/**
 * \brief Basic static function for the interface 'default'.
 * 
 * \param [in] param_BS Interface Parameter "BS".
 * \param [in] instream_biases1 The stream should be of size 256 bytes.
 * \param [in] instream_biases2 The stream should be of size 48 bytes.
 * \param [in] instream_input The stream should be of size (param_BS * 3136) bytes.
 * \param [in] instream_weights1 The stream should be of size 200704 bytes.
 * \param [in] instream_weights2 The stream should be of size 2560 bytes.
 * \param [out] outstream_s1 The stream should be of size (param_BS * 256) bytes.
 * \param [out] outstream_s2 The stream should be of size (param_BS * 40) bytes.
 * \param [out] outstream_x1 The stream should be of size (param_BS * 256) bytes.
 * \param [out] outstream_x2 The stream should be of size (param_BS * 40) bytes.
 * \param [in] routing_string A string containing comma-separated "from_name -> to_name" routing commands.
 */
void FloatForwardProp(
	int64_t param_BS,
	const float *instream_biases1,
	const float *instream_biases2,
	const float *instream_input,
	const float *instream_weights1,
	const float *instream_weights2,
	float *outstream_s1,
	float *outstream_s2,
	float *outstream_x1,
	float *outstream_x2,
	const char * routing_string);

/**
 * \brief Basic static non-blocking function for the interface 'default'.
 * 
 * Schedule to run on an engine and return immediately.
 * The status of the run can be checked either by ::max_wait or ::max_nowait;
 * note that one of these *must* be called, so that associated memory can be released.
 * 
 * 
 * \param [in] param_BS Interface Parameter "BS".
 * \param [in] instream_biases1 The stream should be of size 256 bytes.
 * \param [in] instream_biases2 The stream should be of size 48 bytes.
 * \param [in] instream_input The stream should be of size (param_BS * 3136) bytes.
 * \param [in] instream_weights1 The stream should be of size 200704 bytes.
 * \param [in] instream_weights2 The stream should be of size 2560 bytes.
 * \param [out] outstream_s1 The stream should be of size (param_BS * 256) bytes.
 * \param [out] outstream_s2 The stream should be of size (param_BS * 40) bytes.
 * \param [out] outstream_x1 The stream should be of size (param_BS * 256) bytes.
 * \param [out] outstream_x2 The stream should be of size (param_BS * 40) bytes.
 * \param [in] routing_string A string containing comma-separated "from_name -> to_name" routing commands.
 * \return A handle on the execution status, or NULL in case of error.
 */
max_run_t *FloatForwardProp_nonblock(
	int64_t param_BS,
	const float *instream_biases1,
	const float *instream_biases2,
	const float *instream_input,
	const float *instream_weights1,
	const float *instream_weights2,
	float *outstream_s1,
	float *outstream_s2,
	float *outstream_x1,
	float *outstream_x2,
	const char * routing_string);

/**
 * \brief Advanced static interface, structure for the engine interface 'default'
 * 
 */
typedef struct { 
	int64_t param_BS; /**<  [in] Interface Parameter "BS". */
	const float *instream_biases1; /**<  [in] The stream should be of size 256 bytes. */
	const float *instream_biases2; /**<  [in] The stream should be of size 48 bytes. */
	const float *instream_input; /**<  [in] The stream should be of size (param_BS * 3136) bytes. */
	const float *instream_weights1; /**<  [in] The stream should be of size 200704 bytes. */
	const float *instream_weights2; /**<  [in] The stream should be of size 2560 bytes. */
	float *outstream_s1; /**<  [out] The stream should be of size (param_BS * 256) bytes. */
	float *outstream_s2; /**<  [out] The stream should be of size (param_BS * 40) bytes. */
	float *outstream_x1; /**<  [out] The stream should be of size (param_BS * 256) bytes. */
	float *outstream_x2; /**<  [out] The stream should be of size (param_BS * 40) bytes. */
	const char * routing_string; /**<  [in] A string containing comma-separated "from_name -> to_name" routing commands. */
} FloatForwardProp_actions_t;

/**
 * \brief Advanced static function for the interface 'default'.
 * 
 * \param [in] engine The engine on which the actions will be executed.
 * \param [in,out] interface_actions Actions to be executed.
 */
void FloatForwardProp_run(
	max_engine_t *engine,
	FloatForwardProp_actions_t *interface_actions);

/**
 * \brief Advanced static non-blocking function for the interface 'default'.
 *
 * Schedule the actions to run on the engine and return immediately.
 * The status of the run can be checked either by ::max_wait or ::max_nowait;
 * note that one of these *must* be called, so that associated memory can be released.
 *
 * 
 * \param [in] engine The engine on which the actions will be executed.
 * \param [in] interface_actions Actions to be executed.
 * \return A handle on the execution status of the actions, or NULL in case of error.
 */
max_run_t *FloatForwardProp_run_nonblock(
	max_engine_t *engine,
	FloatForwardProp_actions_t *interface_actions);

/**
 * \brief Group run advanced static function for the interface 'default'.
 * 
 * \param [in] group Group to use.
 * \param [in,out] interface_actions Actions to run.
 *
 * Run the actions on the first device available in the group.
 */
void FloatForwardProp_run_group(max_group_t *group, FloatForwardProp_actions_t *interface_actions);

/**
 * \brief Group run advanced static non-blocking function for the interface 'default'.
 * 
 *
 * Schedule the actions to run on the first device available in the group and return immediately.
 * The status of the run must be checked with ::max_wait. 
 * Note that use of ::max_nowait is prohibited with non-blocking running on groups:
 * see the ::max_run_group_nonblock documentation for more explanation.
 *
 * \param [in] group Group to use.
 * \param [in] interface_actions Actions to run.
 * \return A handle on the execution status of the actions, or NULL in case of error.
 */
max_run_t *FloatForwardProp_run_group_nonblock(max_group_t *group, FloatForwardProp_actions_t *interface_actions);

/**
 * \brief Array run advanced static function for the interface 'default'.
 * 
 * \param [in] engarray The array of devices to use.
 * \param [in,out] interface_actions The array of actions to run.
 *
 * Run the array of actions on the array of engines.  The length of interface_actions
 * must match the size of engarray.
 */
void FloatForwardProp_run_array(max_engarray_t *engarray, FloatForwardProp_actions_t *interface_actions[]);

/**
 * \brief Array run advanced static non-blocking function for the interface 'default'.
 * 
 *
 * Schedule to run the array of actions on the array of engines, and return immediately.
 * The length of interface_actions must match the size of engarray.
 * The status of the run can be checked either by ::max_wait or ::max_nowait;
 * note that one of these *must* be called, so that associated memory can be released.
 *
 * \param [in] engarray The array of devices to use.
 * \param [in] interface_actions The array of actions to run.
 * \return A handle on the execution status of the actions, or NULL in case of error.
 */
max_run_t *FloatForwardProp_run_array_nonblock(max_engarray_t *engarray, FloatForwardProp_actions_t *interface_actions[]);

/**
 * \brief Converts a static-interface action struct into a dynamic-interface max_actions_t struct.
 *
 * Note that this is an internal utility function used by other functions in the static interface.
 *
 * \param [in] maxfile The maxfile to use.
 * \param [in] interface_actions The interface-specific actions to run.
 * \return The dynamic-interface actions to run, or NULL in case of error.
 */
max_actions_t* FloatForwardProp_convert(max_file_t *maxfile, FloatForwardProp_actions_t *interface_actions);

/**
 * \brief Initialise a maxfile.
 */
max_file_t* FloatForwardProp_init(void);

/* Error handling functions */
int FloatForwardProp_has_errors(void);
const char* FloatForwardProp_get_errors(void);
void FloatForwardProp_clear_errors(void);
/* Free statically allocated maxfile data */
void FloatForwardProp_free(void);
/* returns: -1 = error running command; 0 = no error reported */
int FloatForwardProp_simulator_start(void);
/* returns: -1 = error running command; 0 = no error reported */
int FloatForwardProp_simulator_stop(void);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* SLIC_DECLARATIONS_FloatForwardProp_H */
