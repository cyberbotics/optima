/**\file */
#ifndef SLIC_DECLARATIONS_ConvNetwork_H
#define SLIC_DECLARATIONS_ConvNetwork_H
#include "MaxSLiCInterface.h"
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define ConvNetwork_OUT_PAD (2)
#define ConvNetwork_WT_POOL1 (318)
#define ConvNetwork_DYNAMIC_CLOCKS_ENABLED (0)
#define ConvNetwork_WT_POOL2 (157)
#define ConvNetwork_WT_POOL3 (76)
#define ConvNetwork_IS_LINEAR1 (304)
#define ConvNetwork_PAD_LINEAR2 (8)
#define ConvNetwork_OS_LINEAR1 (500)
#define ConvNetwork_PAD_LINEAR1 (256)
#define ConvNetwork_OS_LINEAR2 (2)
#define ConvNetwork_WT_CONV1 (320)
#define ConvNetwork_HT_CONV2 (39)
#define ConvNetwork_HT_CONV1 (80)
#define ConvNetwork_HT_CONV4 (8)
#define ConvNetwork_HT_CONV3 (18)
#define ConvNetwork_WT_CONV4 (38)
#define ConvNetwork_WT_CONV3 (78)
#define ConvNetwork_WT_CONV2 (159)
#define ConvNetwork_IS_LINEAR2 (500)
#define ConvNetwork_K_SIZE (3)
#define ConvNetwork_VS_LINEAR1 (64)
#define ConvNetwork_VS_LINEAR2 (1)
#define ConvNetwork_HT_POOL1 (78)
#define ConvNetwork_HT_POOL2 (37)
#define ConvNetwork_HT_POOL3 (16)
#define ConvNetwork_WINDOW (9)
#define ConvNetwork_IC_CONV1 (3)
#define ConvNetwork_IC_CONV2 (16)
#define ConvNetwork_IC_CONV3 (32)
#define ConvNetwork_OC_CONV3 (64)
#define ConvNetwork_OC_CONV2 (32)
#define ConvNetwork_OC_CONV1 (16)


/*----------------------------------------------------------------------------*/
/*--------------------------- Interface writeLMem ----------------------------*/
/*----------------------------------------------------------------------------*/




/**
 * \brief Basic static function for the interface 'writeLMem'.
 * 
 * \param [in] param_size Interface Parameter "size".
 * \param [in] param_start Interface Parameter "start".
 * \param [in] instream_fromcpu The stream should be of size (param_size * 4) bytes.
 */
void ConvNetwork_writeLMem(
	int32_t param_size,
	int32_t param_start,
	const int32_t *instream_fromcpu);

/**
 * \brief Basic static non-blocking function for the interface 'writeLMem'.
 * 
 * Schedule to run on an engine and return immediately.
 * The status of the run can be checked either by ::max_wait or ::max_nowait;
 * note that one of these *must* be called, so that associated memory can be released.
 * 
 * 
 * \param [in] param_size Interface Parameter "size".
 * \param [in] param_start Interface Parameter "start".
 * \param [in] instream_fromcpu The stream should be of size (param_size * 4) bytes.
 * \return A handle on the execution status, or NULL in case of error.
 */
max_run_t *ConvNetwork_writeLMem_nonblock(
	int32_t param_size,
	int32_t param_start,
	const int32_t *instream_fromcpu);

/**
 * \brief Advanced static interface, structure for the engine interface 'writeLMem'
 * 
 */
typedef struct { 
	int32_t param_size; /**<  [in] Interface Parameter "size". */
	int32_t param_start; /**<  [in] Interface Parameter "start". */
	const int32_t *instream_fromcpu; /**<  [in] The stream should be of size (param_size * 4) bytes. */
} ConvNetwork_writeLMem_actions_t;

/**
 * \brief Advanced static function for the interface 'writeLMem'.
 * 
 * \param [in] engine The engine on which the actions will be executed.
 * \param [in,out] interface_actions Actions to be executed.
 */
void ConvNetwork_writeLMem_run(
	max_engine_t *engine,
	ConvNetwork_writeLMem_actions_t *interface_actions);

/**
 * \brief Advanced static non-blocking function for the interface 'writeLMem'.
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
max_run_t *ConvNetwork_writeLMem_run_nonblock(
	max_engine_t *engine,
	ConvNetwork_writeLMem_actions_t *interface_actions);

/**
 * \brief Group run advanced static function for the interface 'writeLMem'.
 * 
 * \param [in] group Group to use.
 * \param [in,out] interface_actions Actions to run.
 *
 * Run the actions on the first device available in the group.
 */
void ConvNetwork_writeLMem_run_group(max_group_t *group, ConvNetwork_writeLMem_actions_t *interface_actions);

/**
 * \brief Group run advanced static non-blocking function for the interface 'writeLMem'.
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
max_run_t *ConvNetwork_writeLMem_run_group_nonblock(max_group_t *group, ConvNetwork_writeLMem_actions_t *interface_actions);

/**
 * \brief Array run advanced static function for the interface 'writeLMem'.
 * 
 * \param [in] engarray The array of devices to use.
 * \param [in,out] interface_actions The array of actions to run.
 *
 * Run the array of actions on the array of engines.  The length of interface_actions
 * must match the size of engarray.
 */
void ConvNetwork_writeLMem_run_array(max_engarray_t *engarray, ConvNetwork_writeLMem_actions_t *interface_actions[]);

/**
 * \brief Array run advanced static non-blocking function for the interface 'writeLMem'.
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
max_run_t *ConvNetwork_writeLMem_run_array_nonblock(max_engarray_t *engarray, ConvNetwork_writeLMem_actions_t *interface_actions[]);

/**
 * \brief Converts a static-interface action struct into a dynamic-interface max_actions_t struct.
 *
 * Note that this is an internal utility function used by other functions in the static interface.
 *
 * \param [in] maxfile The maxfile to use.
 * \param [in] interface_actions The interface-specific actions to run.
 * \return The dynamic-interface actions to run, or NULL in case of error.
 */
max_actions_t* ConvNetwork_writeLMem_convert(max_file_t *maxfile, ConvNetwork_writeLMem_actions_t *interface_actions);



/*----------------------------------------------------------------------------*/
/*---------------------------- Interface default -----------------------------*/
/*----------------------------------------------------------------------------*/




/**
 * \brief Basic static function for the interface 'default'.
 * 
 * \param [in] instream_inputfromcpu The stream should be of size 307200 bytes.
 * \param [out] outstream_outputtocpu The stream should be of size 16 bytes.
 * \param [in] inmem_CONVOLUTION_LAYER1_biasMem Mapped ROM inmem_CONVOLUTION_LAYER1_biasMem, should be of size (17 * sizeof(double)).
 * \param [in] inmem_CONVOLUTION_LAYER2_biasMem Mapped ROM inmem_CONVOLUTION_LAYER2_biasMem, should be of size (33 * sizeof(double)).
 * \param [in] inmem_CONVOLUTION_LAYER3_biasMem Mapped ROM inmem_CONVOLUTION_LAYER3_biasMem, should be of size (65 * sizeof(double)).
 * \param [in] inmem_LINEAR_LAYER1_biasMem Mapped ROM inmem_LINEAR_LAYER1_biasMem, should be of size (501 * sizeof(double)).
 * \param [in] inmem_LINEAR_LAYER2_biasMem Mapped ROM inmem_LINEAR_LAYER2_biasMem, should be of size (3 * sizeof(double)).
 */
void ConvNetwork(
	const int32_t *instream_inputfromcpu,
	int32_t *outstream_outputtocpu,
	const double *inmem_CONVOLUTION_LAYER1_biasMem,
	const double *inmem_CONVOLUTION_LAYER2_biasMem,
	const double *inmem_CONVOLUTION_LAYER3_biasMem,
	const double *inmem_LINEAR_LAYER1_biasMem,
	const double *inmem_LINEAR_LAYER2_biasMem);

/**
 * \brief Basic static non-blocking function for the interface 'default'.
 * 
 * Schedule to run on an engine and return immediately.
 * The status of the run can be checked either by ::max_wait or ::max_nowait;
 * note that one of these *must* be called, so that associated memory can be released.
 * 
 * 
 * \param [in] instream_inputfromcpu The stream should be of size 307200 bytes.
 * \param [out] outstream_outputtocpu The stream should be of size 16 bytes.
 * \param [in] inmem_CONVOLUTION_LAYER1_biasMem Mapped ROM inmem_CONVOLUTION_LAYER1_biasMem, should be of size (17 * sizeof(double)).
 * \param [in] inmem_CONVOLUTION_LAYER2_biasMem Mapped ROM inmem_CONVOLUTION_LAYER2_biasMem, should be of size (33 * sizeof(double)).
 * \param [in] inmem_CONVOLUTION_LAYER3_biasMem Mapped ROM inmem_CONVOLUTION_LAYER3_biasMem, should be of size (65 * sizeof(double)).
 * \param [in] inmem_LINEAR_LAYER1_biasMem Mapped ROM inmem_LINEAR_LAYER1_biasMem, should be of size (501 * sizeof(double)).
 * \param [in] inmem_LINEAR_LAYER2_biasMem Mapped ROM inmem_LINEAR_LAYER2_biasMem, should be of size (3 * sizeof(double)).
 * \return A handle on the execution status, or NULL in case of error.
 */
max_run_t *ConvNetwork_nonblock(
	const int32_t *instream_inputfromcpu,
	int32_t *outstream_outputtocpu,
	const double *inmem_CONVOLUTION_LAYER1_biasMem,
	const double *inmem_CONVOLUTION_LAYER2_biasMem,
	const double *inmem_CONVOLUTION_LAYER3_biasMem,
	const double *inmem_LINEAR_LAYER1_biasMem,
	const double *inmem_LINEAR_LAYER2_biasMem);

/**
 * \brief Advanced static interface, structure for the engine interface 'default'
 * 
 */
typedef struct { 
	const int32_t *instream_inputfromcpu; /**<  [in] The stream should be of size 307200 bytes. */
	int32_t *outstream_outputtocpu; /**<  [out] The stream should be of size 16 bytes. */
	const double *inmem_CONVOLUTION_LAYER1_biasMem; /**<  [in] Mapped ROM inmem_CONVOLUTION_LAYER1_biasMem, should be of size (17 * sizeof(double)). */
	const double *inmem_CONVOLUTION_LAYER2_biasMem; /**<  [in] Mapped ROM inmem_CONVOLUTION_LAYER2_biasMem, should be of size (33 * sizeof(double)). */
	const double *inmem_CONVOLUTION_LAYER3_biasMem; /**<  [in] Mapped ROM inmem_CONVOLUTION_LAYER3_biasMem, should be of size (65 * sizeof(double)). */
	const double *inmem_LINEAR_LAYER1_biasMem; /**<  [in] Mapped ROM inmem_LINEAR_LAYER1_biasMem, should be of size (501 * sizeof(double)). */
	const double *inmem_LINEAR_LAYER2_biasMem; /**<  [in] Mapped ROM inmem_LINEAR_LAYER2_biasMem, should be of size (3 * sizeof(double)). */
} ConvNetwork_actions_t;

/**
 * \brief Advanced static function for the interface 'default'.
 * 
 * \param [in] engine The engine on which the actions will be executed.
 * \param [in,out] interface_actions Actions to be executed.
 */
void ConvNetwork_run(
	max_engine_t *engine,
	ConvNetwork_actions_t *interface_actions);

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
max_run_t *ConvNetwork_run_nonblock(
	max_engine_t *engine,
	ConvNetwork_actions_t *interface_actions);

/**
 * \brief Group run advanced static function for the interface 'default'.
 * 
 * \param [in] group Group to use.
 * \param [in,out] interface_actions Actions to run.
 *
 * Run the actions on the first device available in the group.
 */
void ConvNetwork_run_group(max_group_t *group, ConvNetwork_actions_t *interface_actions);

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
max_run_t *ConvNetwork_run_group_nonblock(max_group_t *group, ConvNetwork_actions_t *interface_actions);

/**
 * \brief Array run advanced static function for the interface 'default'.
 * 
 * \param [in] engarray The array of devices to use.
 * \param [in,out] interface_actions The array of actions to run.
 *
 * Run the array of actions on the array of engines.  The length of interface_actions
 * must match the size of engarray.
 */
void ConvNetwork_run_array(max_engarray_t *engarray, ConvNetwork_actions_t *interface_actions[]);

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
max_run_t *ConvNetwork_run_array_nonblock(max_engarray_t *engarray, ConvNetwork_actions_t *interface_actions[]);

/**
 * \brief Converts a static-interface action struct into a dynamic-interface max_actions_t struct.
 *
 * Note that this is an internal utility function used by other functions in the static interface.
 *
 * \param [in] maxfile The maxfile to use.
 * \param [in] interface_actions The interface-specific actions to run.
 * \return The dynamic-interface actions to run, or NULL in case of error.
 */
max_actions_t* ConvNetwork_convert(max_file_t *maxfile, ConvNetwork_actions_t *interface_actions);

/**
 * \brief Initialise a maxfile.
 */
max_file_t* ConvNetwork_init(void);

/* Error handling functions */
int ConvNetwork_has_errors(void);
const char* ConvNetwork_get_errors(void);
void ConvNetwork_clear_errors(void);
/* Free statically allocated maxfile data */
void ConvNetwork_free(void);
/* These are dummy functions for hardware builds. */
int ConvNetwork_simulator_start(void);
int ConvNetwork_simulator_stop(void);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* SLIC_DECLARATIONS_ConvNetwork_H */
