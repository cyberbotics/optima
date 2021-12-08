/**\file */
#ifndef SLIC_DECLARATIONS_TileForwardProp_H
#define SLIC_DECLARATIONS_TileForwardProp_H
#include "MaxSLiCInterface.h"
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define TileForwardProp_SIZE_LAYER_2 (12)
#define TileForwardProp_SIZE_LAYER_0 (784)
#define TileForwardProp_DYNAMIC_CLOCKS_ENABLED (0)
#define TileForwardProp_SIZE_LAYER_1 (64)
#define TileForwardProp_PCIE_ALIGNMENT (16)
#define TileForwardProp_IN_VEC_SIZE1 (28)
#define TileForwardProp_IN_VEC_SIZE2 (32)
#define TileForwardProp_TILE_OFFSET1 (16)
#define TileForwardProp_TILE_OFFSET2 (12)


/*----------------------------------------------------------------------------*/
/*---------------------------- Interface default -----------------------------*/
/*----------------------------------------------------------------------------*/




/**
 * \brief Basic static function for the interface 'default'.
 * 
 * \param [in] param_BS Interface Parameter "BS".
 * \param [in] instream_biases1 The stream should be of size 256 bytes.
 * \param [in] instream_biases2 The stream should be of size 56 bytes.
 * \param [in] instream_input The stream should be of size (param_BS * 3136) bytes.
 * \param [in] instream_weights1 The stream should be of size 200704 bytes.
 * \param [in] instream_weights2 The stream should be of size 3072 bytes.
 * \param [out] outstream_s1 The stream should be of size (param_BS * 256) bytes.
 * \param [out] outstream_s2 The stream should be of size (param_BS * 48) bytes.
 * \param [out] outstream_x1 The stream should be of size (param_BS * 256) bytes.
 * \param [out] outstream_x2 The stream should be of size (param_BS * 48) bytes.
 * \param [in] routing_string A string containing comma-separated "from_name -> to_name" routing commands.
 */
void TileForwardProp(
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
 * \param [in] instream_biases2 The stream should be of size 56 bytes.
 * \param [in] instream_input The stream should be of size (param_BS * 3136) bytes.
 * \param [in] instream_weights1 The stream should be of size 200704 bytes.
 * \param [in] instream_weights2 The stream should be of size 3072 bytes.
 * \param [out] outstream_s1 The stream should be of size (param_BS * 256) bytes.
 * \param [out] outstream_s2 The stream should be of size (param_BS * 48) bytes.
 * \param [out] outstream_x1 The stream should be of size (param_BS * 256) bytes.
 * \param [out] outstream_x2 The stream should be of size (param_BS * 48) bytes.
 * \param [in] routing_string A string containing comma-separated "from_name -> to_name" routing commands.
 * \return A handle on the execution status, or NULL in case of error.
 */
max_run_t *TileForwardProp_nonblock(
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
	const float *instream_biases2; /**<  [in] The stream should be of size 56 bytes. */
	const float *instream_input; /**<  [in] The stream should be of size (param_BS * 3136) bytes. */
	const float *instream_weights1; /**<  [in] The stream should be of size 200704 bytes. */
	const float *instream_weights2; /**<  [in] The stream should be of size 3072 bytes. */
	float *outstream_s1; /**<  [out] The stream should be of size (param_BS * 256) bytes. */
	float *outstream_s2; /**<  [out] The stream should be of size (param_BS * 48) bytes. */
	float *outstream_x1; /**<  [out] The stream should be of size (param_BS * 256) bytes. */
	float *outstream_x2; /**<  [out] The stream should be of size (param_BS * 48) bytes. */
	const char * routing_string; /**<  [in] A string containing comma-separated "from_name -> to_name" routing commands. */
} TileForwardProp_actions_t;

/**
 * \brief Advanced static function for the interface 'default'.
 * 
 * \param [in] engine The engine on which the actions will be executed.
 * \param [in,out] interface_actions Actions to be executed.
 */
void TileForwardProp_run(
	max_engine_t *engine,
	TileForwardProp_actions_t *interface_actions);

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
max_run_t *TileForwardProp_run_nonblock(
	max_engine_t *engine,
	TileForwardProp_actions_t *interface_actions);

/**
 * \brief Group run advanced static function for the interface 'default'.
 * 
 * \param [in] group Group to use.
 * \param [in,out] interface_actions Actions to run.
 *
 * Run the actions on the first device available in the group.
 */
void TileForwardProp_run_group(max_group_t *group, TileForwardProp_actions_t *interface_actions);

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
max_run_t *TileForwardProp_run_group_nonblock(max_group_t *group, TileForwardProp_actions_t *interface_actions);

/**
 * \brief Array run advanced static function for the interface 'default'.
 * 
 * \param [in] engarray The array of devices to use.
 * \param [in,out] interface_actions The array of actions to run.
 *
 * Run the array of actions on the array of engines.  The length of interface_actions
 * must match the size of engarray.
 */
void TileForwardProp_run_array(max_engarray_t *engarray, TileForwardProp_actions_t *interface_actions[]);

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
max_run_t *TileForwardProp_run_array_nonblock(max_engarray_t *engarray, TileForwardProp_actions_t *interface_actions[]);

/**
 * \brief Converts a static-interface action struct into a dynamic-interface max_actions_t struct.
 *
 * Note that this is an internal utility function used by other functions in the static interface.
 *
 * \param [in] maxfile The maxfile to use.
 * \param [in] interface_actions The interface-specific actions to run.
 * \return The dynamic-interface actions to run, or NULL in case of error.
 */
max_actions_t* TileForwardProp_convert(max_file_t *maxfile, TileForwardProp_actions_t *interface_actions);

/**
 * \brief Initialise a maxfile.
 */
max_file_t* TileForwardProp_init(void);

/* Error handling functions */
int TileForwardProp_has_errors(void);
const char* TileForwardProp_get_errors(void);
void TileForwardProp_clear_errors(void);
/* Free statically allocated maxfile data */
void TileForwardProp_free(void);
/* returns: -1 = error running command; 0 = no error reported */
int TileForwardProp_simulator_start(void);
/* returns: -1 = error running command; 0 = no error reported */
int TileForwardProp_simulator_stop(void);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* SLIC_DECLARATIONS_TileForwardProp_H */
