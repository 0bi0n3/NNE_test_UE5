#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"

#include "NNECore.h"
#include "NNECoreRuntimeCPU.h"
#include "NNECoreModelData.h"

#include "NeuralNetworkModel.generated.h"

USTRUCT(BlueprintType, Category = "NNE - Tutorial")
struct FNeuralNetworkTensor
{
	GENERATED_BODY()

public:

	UPROPERTY(BlueprintReadWrite, Category = "NNE - Tutorial")
	TArray<int32> Shape = TArray<int32>();

	UPROPERTY(BlueprintReadWrite, Category = "NNE - Tutorial")
	TArray<float> Data = TArray<float>();
};

UCLASS(BlueprintType, Category = "NNE - Tutorial")
class NNETUTORIAL_API UNeuralNetworkModel : public UObject
{
	GENERATED_BODY()

public:

	UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
	static TArray<FString> GetRuntimeNames();

	UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
	static UNeuralNetworkModel* CreateModel(UObject* Parent, FString RuntimeName, UNNEModelData* ModelData);

	UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
	static bool CreateTensor(TArray<int32> Shape, UPARAM(ref) FNeuralNetworkTensor& Tensor);

public:

	UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
	int32 NumInputs();

	UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
	int32 NumOutputs();

	UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
	TArray<int32> GetInputShape(int32 Index);

	UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
	TArray<int32> GetOutputShape(int32 Index);

public:

	UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
	bool SetInputs(const TArray<FNeuralNetworkTensor>& Inputs);

	UFUNCTION(BlueprintCallable, Category = "NNE - Tutorial")
	bool RunSync(UPARAM(ref) TArray<FNeuralNetworkTensor>& Outputs);

private:

	TSharedPtr<UE::NNECore::IModelCPU> Model;

	TArray<UE::NNECore::FTensorBindingCPU> InputBindings;
	TArray<UE::NNECore::FTensorShape> InputShapes;
};