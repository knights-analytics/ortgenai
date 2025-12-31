#ifndef ORT_GENAI_WRAPPER_H
#define ORT_GENAI_WRAPPER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#undef _WIN32
#include "ort_genai_c.h"

// ... However, mingw will complain if _WIN32 is *not* defined! So redefine it.
#define _WIN32

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OgaResult OgaResult;
typedef struct OgaModel OgaModel;
typedef struct OgaTokenizer OgaTokenizer;
typedef struct OgaTokenizerStream OgaTokenizerStream;
typedef struct OgaSequences OgaSequences;
typedef struct OgaGenerator OgaGenerator;
typedef struct OgaGeneratorParams OgaGeneratorParams;
typedef struct OgaConfig OgaConfig;
typedef struct OgaImages OgaImages;
typedef struct OgaMultiModalProcessor OgaMultiModalProcessor;
typedef struct OgaNamedTensors OgaNamedTensors;
typedef struct OgaStringArray OgaStringArray;

// Function pointer typedefs for the subset of GenAI C API we wrap.
typedef OgaResult* (*PFN_OgaCreateModel)(const char*, OgaModel**);
typedef OgaResult* (*PFN_OgaCreateTokenizer)(const OgaModel*, OgaTokenizer**);
typedef OgaResult* (*PFN_OgaCreateTokenizerStream)(const OgaTokenizer*, OgaTokenizerStream**);
typedef const char* (*PFN_OgaResultGetError)(const OgaResult*);
typedef void (*PFN_OgaDestroyResult)(OgaResult*);
typedef void (*PFN_OgaDestroyModel)(OgaModel*);
typedef void (*PFN_OgaDestroyTokenizer)(OgaTokenizer*);
typedef void (*PFN_OgaDestroyTokenizerStream)(OgaTokenizerStream*);
typedef OgaResult* (*PFN_OgaTokenizerApplyChatTemplate)(const OgaTokenizer*, const char*, const char*, const char*, bool, const char**);
typedef void (*PFN_OgaDestroyString)(const char*);
typedef OgaResult* (*PFN_OgaCreateSequences)(OgaSequences**);
typedef void (*PFN_OgaDestroySequences)(OgaSequences*);
typedef OgaResult* (*PFN_OgaTokenizerEncode)(const OgaTokenizer*, const char*, OgaSequences*);
typedef OgaResult* (*PFN_OgaCreateGenerator)(const OgaModel*, const OgaGeneratorParams*, OgaGenerator**);
typedef void (*PFN_OgaDestroyGenerator)(OgaGenerator*);
typedef OgaResult* (*PFN_OgaCreateGeneratorParams)(const OgaModel*, OgaGeneratorParams**);
typedef void (*PFN_OgaDestroyGeneratorParams)(OgaGeneratorParams*);
typedef OgaResult* (*PFN_OgaGeneratorParamsSetSearchNumber)(OgaGeneratorParams*, const char*, double);
typedef OgaResult* (*PFN_OgaGeneratorAppendTokenSequences)(OgaGenerator*, const OgaSequences*);
typedef OgaResult* (*PFN_OgaGeneratorSetInputs)(OgaGenerator*, const OgaNamedTensors*);
typedef OgaResult* (*PFN_OgaGeneratorGenerateNextToken)(OgaGenerator*);
typedef size_t (*PFN_OgaGeneratorGetSequenceCount)(const OgaGenerator*, size_t);
typedef const int32_t* (*PFN_OgaGeneratorGetSequenceData)(const OgaGenerator*, size_t);
typedef OgaResult* (*PFN_OgaTokenizerStreamDecode)(OgaTokenizerStream*, int32_t, const char**);
typedef bool (*PFN_OgaGeneratorIsDone)(const OgaGenerator*);

// Config-related API
typedef OgaResult* (*PFN_OgaCreateConfig)(const char*, OgaConfig**);
typedef OgaResult* (*PFN_OgaConfigClearProviders)(OgaConfig*);
typedef OgaResult* (*PFN_OgaConfigAppendProvider)(OgaConfig*, const char*);
typedef OgaResult* (*PFN_OgaConfigSetProviderOption)(OgaConfig*, const char*, const char*, const char*);
typedef OgaResult* (*PFN_OgaCreateModelFromConfig)(const OgaConfig*, OgaModel**);

// Multimodal API
typedef OgaResult* (*PFN_OgaLoadImage)(const char*, OgaImages**);
typedef OgaResult* (*PFN_OgaLoadImages)(const OgaStringArray*, OgaImages**);
typedef OgaResult* (*PFN_OgaLoadImagesFromBuffers)(const void**, const size_t*, size_t, OgaImages**);
typedef void (*PFN_OgaDestroyImages)(OgaImages*);
typedef OgaResult* (*PFN_OgaCreateMultiModalProcessor)(const OgaModel*, OgaMultiModalProcessor**);
typedef void (*PFN_OgaDestroyMultiModalProcessor)(OgaMultiModalProcessor*);
typedef OgaResult* (*PFN_OgaProcessorProcessImages)(const OgaMultiModalProcessor*, const char*, const OgaImages*, OgaNamedTensors**);
typedef void (*PFN_OgaDestroyNamedTensors)(OgaNamedTensors*);
typedef OgaResult* (*PFN_OgaCreateStringArray)(OgaStringArray**);
typedef void (*PFN_OgaDestroyStringArray)(OgaStringArray*);
typedef OgaResult* (*PFN_OgaStringArrayAddString)(OgaStringArray*, const char*);

// Aggregated API table mirroring the pattern used by OrtApi in onnxruntime_go.
typedef struct GenAiApiTable {
	PFN_OgaCreateModel        CreateModel;
	PFN_OgaResultGetError     ResultGetError;
	PFN_OgaDestroyResult      DestroyResult;
	PFN_OgaDestroyModel       DestroyModel;
	PFN_OgaCreateTokenizer    CreateTokenizer;
	PFN_OgaDestroyTokenizer   DestroyTokenizer;
	PFN_OgaCreateTokenizerStream CreateTokenizerStream;
	PFN_OgaDestroyTokenizerStream DestroyTokenizerStream;
	PFN_OgaTokenizerApplyChatTemplate  ApplyChatTemplate;
	PFN_OgaDestroyString DestroyString;
	PFN_OgaCreateSequences CreateSequences;
	PFN_OgaDestroySequences DestroySequences;
	PFN_OgaTokenizerEncode TokenizerEncode;
	PFN_OgaCreateGenerator CreateGenerator;
	PFN_OgaDestroyGenerator DestroyGenerator;
	PFN_OgaCreateGeneratorParams CreateGeneratorParams;
	PFN_OgaDestroyGeneratorParams DestroyGeneratorParams;
	PFN_OgaGeneratorParamsSetSearchNumber GeneratorParamsSetSearchNumber;
	PFN_OgaGeneratorAppendTokenSequences GeneratorAppendTokenSequences;
	PFN_OgaGeneratorSetInputs GeneratorSetInputs;
	PFN_OgaGeneratorGenerateNextToken GeneratorGenerateNextToken;
	PFN_OgaGeneratorGetSequenceCount GeneratorGetSequenceCount;
	PFN_OgaGeneratorGetSequenceData GeneratorGetSequenceData;
	PFN_OgaTokenizerStreamDecode TokenizerStreamDecode;
	PFN_OgaGeneratorIsDone IsDone;
	// Config
	PFN_OgaCreateConfig CreateConfig;
	PFN_OgaConfigClearProviders ConfigClearProviders;
	PFN_OgaConfigAppendProvider ConfigAppendProvider;
	PFN_OgaConfigSetProviderOption ConfigSetProviderOption;
	PFN_OgaCreateModelFromConfig CreateModelFromConfig;
	// Multimodal
	PFN_OgaLoadImage LoadImage;
	PFN_OgaLoadImages LoadImages;
	PFN_OgaLoadImagesFromBuffers LoadImagesFromBuffers;
	PFN_OgaDestroyImages DestroyImages;
	PFN_OgaCreateMultiModalProcessor CreateMultiModalProcessor;
	PFN_OgaDestroyMultiModalProcessor DestroyMultiModalProcessor;
	PFN_OgaProcessorProcessImages ProcessorProcessImages;
	PFN_OgaDestroyNamedTensors DestroyNamedTensors;
	PFN_OgaCreateStringArray CreateStringArray;
	PFN_OgaDestroyStringArray DestroyStringArray;
	PFN_OgaStringArrayAddString StringArrayAddString;
} GenAiApiTable;

// Sets the global function pointer table. All pointers must be non-null.
// Returns 0 on success, non-zero on failure.
int SetGenAiApi(void* createModel,
				void* resultGetError,
				void* destroyResult,
				void* destroyModel,
				void* createTokenizer,
				void* destroyTokenizer,
				void* createTokenizerStream,
				void* destroyTokenizerStream,
				void* applyChatTemplate,
				void* destroyString,
				void* createSequences,
				void* destroySequences,
			void* tokenizerEncode,
			void* createGenerator,
			void* destroyGenerator,
			void* createGeneratorParams,
			void* destroyGeneratorParams,
		void* generatorParamsSetSearchNumber,
	void* generatorAppendTokenSequences,
	void* generatorSetInputs,
	void* generatorGenerateNextToken,
	void* generatorGetSequenceCount,
	void* generatorGetSequenceData,
	void* tokenizerStreamDecode,
	void* isDone,
	// Config
	void* createConfig,
	void* configClearProviders,
	void* configAppendProvider,
	void* configSetProviderOption,
	void* createModelFromConfig,
	// Multimodal
	void* loadImage,
	void* loadImages,
	void* loadImagesFromBuffers,
	void* destroyImages,
	void* createMultiModalProcessor,
	void* destroyMultiModalProcessor,
	void* processorProcessImages,
	void* destroyNamedTensors,
	void* createStringArray,
	void* destroyStringArray,
	void* stringArrayAddString);

// Returns non-zero if the API table is initialized.
int GenAiApiIsInitialized(void);

// Thin wrappers that call through the function pointer table.
OgaResult* CreateOgaModel(const char* config_path, OgaModel** out);
OgaResult* CreateOgaTokenizer(const OgaModel* model, OgaTokenizer** out);
OgaResult* CreateOgaTokenizerStream(const OgaTokenizer* tokenizer, OgaTokenizerStream** out);
OgaResult* CreateOgaSequences(OgaSequences** out);
OgaResult* CreateOgaGenerator(const OgaModel* model, const OgaGeneratorParams* generatorParams, OgaGenerator** out);
OgaResult* CreateOgaGeneratorParams(const OgaModel* model,OgaGeneratorParams** out);

const char* GetOgaResultErrorString(const OgaResult* result);
void DestroyOgaResult(OgaResult* result);
void DestroyOgaModel(OgaModel* model);
void DestroyOgaTokenizer(OgaTokenizer* tokenizer);
void DestroyOgaTokenizerStream(OgaTokenizerStream* tokenizerStream);
void DestroyOgaString(const char*);
void DestroyOgaSequences(OgaSequences* sequences);
void DestroyOgaGenerator(OgaGenerator* generator);
void DestroyOgaGeneratorParams(OgaGeneratorParams* generatorParams);
OgaResult* TokenizerEncode(const OgaTokenizer* tokenizer, const char* str, OgaSequences* sequences);

OgaResult* ApplyOgaTokenizerChatTemplate(const OgaTokenizer* tokenizer, const char* input, const char* param1, const char* param2, bool flag, const char** output);

OgaResult* GeneratorParamsSetSearchNumber(OgaGeneratorParams* generatorParams, const char* name, double searchNumber);
OgaResult* GeneratorAppendTokenSequences(OgaGenerator* generator, OgaSequences* sequences);
OgaResult* GeneratorSetInputs(OgaGenerator* generator, const OgaNamedTensors* named_tensors);
OgaResult* GeneratorGenerateNextToken(OgaGenerator* generator);
size_t GeneratorGetSequenceCount(const OgaGenerator* generator, size_t sequence_index);
const int32_t* GeneratorGetSequenceData(const OgaGenerator* generator, size_t sequence_index);
OgaResult* TokenizerStreamDecode(OgaTokenizerStream* tokenizerStream, int32_t token, const char** output);
bool IsDone(const OgaGenerator* generator);

// Config thin wrappers
OgaResult* CreateOgaConfig(const char* config_path, OgaConfig** out);
OgaResult* OgaConfigClearProviders(OgaConfig* config);
OgaResult* OgaConfigAppendProvider(OgaConfig* config, const char* provider);
OgaResult* OgaConfigSetProviderOption(OgaConfig* config, const char* provider, const char* key, const char* value);
OgaResult* CreateOgaModelFromConfig(const OgaConfig* config, OgaModel** out);

// Multimodal thin wrappers
OgaResult* LoadOgaImage(const char* image_path, OgaImages** out);
OgaResult* LoadOgaImages(const OgaStringArray* image_paths, OgaImages** out);
OgaResult* LoadOgaImagesFromBuffers(const void** image_data, const size_t* image_data_sizes, size_t count, OgaImages** out);
void DestroyOgaImages(OgaImages* images);
OgaResult* CreateOgaMultiModalProcessor(const OgaModel* model, OgaMultiModalProcessor** out);
void DestroyOgaMultiModalProcessor(OgaMultiModalProcessor* processor);
OgaResult* ProcessOgaImages(const OgaMultiModalProcessor* processor, const char* prompt, const OgaImages* images, OgaNamedTensors** out);
void DestroyOgaNamedTensors(OgaNamedTensors* tensors);
OgaResult* CreateOgaStringArray(OgaStringArray** out);
void DestroyOgaStringArray(OgaStringArray* string_array);
OgaResult* AddStringToOgaStringArray(OgaStringArray* string_array, const char* str);

#ifdef __cplusplus
} // extern "C"
#endif

#endif  // ORT_GENAI_WRAPPER_H
