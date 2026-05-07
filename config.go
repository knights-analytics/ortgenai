package ortgenai

/*
#include "ort_genai_wrapper.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type Config struct {
	configPtr *C.OgaConfig
}

func CreateConfig(configPath string) (*Config, error) {
	cPath := C.CString(configPath)
	defer C.free(unsafe.Pointer(cPath))

	var config *C.OgaConfig
	result := C.CreateOgaConfig(cPath, &config)
	if err := OgaResultToError(result); err != nil {
		return nil, fmt.Errorf("CreateConfig failed: %w", err)
	}
	return &Config{configPtr: config}, nil
}

func (c *Config) Destroy() {
	if c.configPtr != nil {
		C.DestroyOgaConfig(c.configPtr)
		c.configPtr = nil
	}
}

func (c *Config) ClearProviders() error {
	result := C.OgaConfigClearProviders(c.configPtr)
	if err := OgaResultToError(result); err != nil {
		return fmt.Errorf("ConfigClearProviders failed: %w", err)
	}
	return nil
}

func (c *Config) AppendProvider(provider string) error {
	cProvider := C.CString(provider)
	defer C.free(unsafe.Pointer(cProvider))
	result := C.OgaConfigAppendProvider(c.configPtr, cProvider)
	if err := OgaResultToError(result); err != nil {
		return fmt.Errorf("ConfigAppendProvider failed: %w", err)
	}
	return nil
}

func (c *Config) SetProviderOption(provider, key, value string) error {
	cProvider := C.CString(provider)
	defer C.free(unsafe.Pointer(cProvider))
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	cValue := C.CString(value)
	defer C.free(unsafe.Pointer(cValue))

	result := C.OgaConfigSetProviderOption(c.configPtr, cProvider, cKey, cValue)
	if err := OgaResultToError(result); err != nil {
		return fmt.Errorf("ConfigSetProviderOption failed: %w", err)
	}
	return nil
}

func (c *Config) AddModelData(modelFilename string, modelData []byte) error {
	cFilename := C.CString(modelFilename)
	defer C.free(unsafe.Pointer(cFilename))

	data := unsafe.Pointer(&modelData[0])
	result := C.OgaConfigAddModelData(c.configPtr, cFilename, data, C.size_t(len(modelData)))
	if err := OgaResultToError(result); err != nil {
		return fmt.Errorf("ConfigAddModelData failed: %w", err)
	}
	return nil
}

func (c *Config) RemoveModelData(modelFilename string) error {
	cFilename := C.CString(modelFilename)
	defer C.free(unsafe.Pointer(cFilename))

	result := C.OgaConfigRemoveModelData(c.configPtr, cFilename)
	if err := OgaResultToError(result); err != nil {
		return fmt.Errorf("ConfigRemoveModelData failed: %w", err)
	}
	return nil
}

func (c *Config) Overlay(json string) error {
	cJson := C.CString(json)
	defer C.free(unsafe.Pointer(cJson))

	result := C.OgaConfigOverlay(c.configPtr, cJson)
	if err := OgaResultToError(result); err != nil {
		return fmt.Errorf("ConfigOverlay failed: %w", err)
	}
	return nil
}

func (c *Config) SetDecoderProviderOptionsHardwareDeviceType(provider, hardwareDeviceType string) error {
	cProvider := C.CString(provider)
	defer C.free(unsafe.Pointer(cProvider))
	cType := C.CString(hardwareDeviceType)
	defer C.free(unsafe.Pointer(cType))

	result := C.OgaConfigSetDecoderProviderOptionsHardwareDeviceType(c.configPtr, cProvider, cType)
	if err := OgaResultToError(result); err != nil {
		return fmt.Errorf("ConfigSetDecoderProviderOptionsHardwareDeviceType failed: %w", err)
	}
	return nil
}

func (c *Config) SetDecoderProviderOptionsHardwareDeviceID(provider string, hardwareDeviceID uint32) error {
	cProvider := C.CString(provider)
	defer C.free(unsafe.Pointer(cProvider))

	result := C.OgaConfigSetDecoderProviderOptionsHardwareDeviceId(c.configPtr, cProvider, C.uint32_t(hardwareDeviceID))
	if err := OgaResultToError(result); err != nil {
		return fmt.Errorf("ConfigSetDecoderProviderOptionsHardwareDeviceID failed: %w", err)
	}
	return nil
}

func (c *Config) SetDecoderProviderOptionsHardwareVendorID(provider string, hardwareVendorID uint32) error {
	cProvider := C.CString(provider)
	defer C.free(unsafe.Pointer(cProvider))

	result := C.OgaConfigSetDecoderProviderOptionsHardwareVendorId(c.configPtr, cProvider, C.uint32_t(hardwareVendorID))
	if err := OgaResultToError(result); err != nil {
		return fmt.Errorf("ConfigSetDecoderProviderOptionsHardwareVendorID failed: %w", err)
	}
	return nil
}

func (c *Config) ClearDecoderProviderOptionsHardwareDeviceType(provider string) error {
	cProvider := C.CString(provider)
	defer C.free(unsafe.Pointer(cProvider))

	result := C.OgaConfigClearDecoderProviderOptionsHardwareDeviceType(c.configPtr, cProvider)
	if err := OgaResultToError(result); err != nil {
		return fmt.Errorf("ConfigClearDecoderProviderOptionsHardwareDeviceType failed: %w", err)
	}
	return nil
}

func (c *Config) ClearDecoderProviderOptionsHardwareDeviceID(provider string) error {
	cProvider := C.CString(provider)
	defer C.free(unsafe.Pointer(cProvider))

	result := C.OgaConfigClearDecoderProviderOptionsHardwareDeviceId(c.configPtr, cProvider)
	if err := OgaResultToError(result); err != nil {
		return fmt.Errorf("ConfigClearDecoderProviderOptionsHardwareDeviceID failed: %w", err)
	}
	return nil
}

func (c *Config) ClearDecoderProviderOptionsHardwareVendorID(provider string) error {
	cProvider := C.CString(provider)
	defer C.free(unsafe.Pointer(cProvider))

	result := C.OgaConfigClearDecoderProviderOptionsHardwareVendorId(c.configPtr, cProvider)
	if err := OgaResultToError(result); err != nil {
		return fmt.Errorf("ConfigClearDecoderProviderOptionsHardwareVendorID failed: %w", err)
	}
	return nil
}

func (c *Config) CreateModel() (*Model, error) {
	var modelPtr *C.OgaModel
	result := C.CreateOgaModelFromConfig(c.configPtr, &modelPtr)
	if err := OgaResultToError(result); err != nil {
		return nil, fmt.Errorf("CreateModelFromConfig failed: %w", err)
	}
	return &Model{modelPtr: modelPtr}, nil
}
