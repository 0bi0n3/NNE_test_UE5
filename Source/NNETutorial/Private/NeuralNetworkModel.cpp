// Fill out your copyright notice in the Description page of Project Settings.


#include "NeuralNetworkModel.h"

TArray<FString> UNeuralNetworkModel::GetRuntimeNames()
{
	using namespace UE::NNECore;

	TArray<FString> Result;
	TArrayView<TWeakInterfacePtr<INNERuntime>> Runtimes = GetAllRuntimes();
	for (int32 i = 0; i < Runtimes.Num(); i++)
	{
		if (Runtimes[i].IsValid() && Cast<INNERuntimeCPU>(Runtimes[i].Get()))
		{
			Result.Add(Runtimes[i]->GetRuntimeName());
		}
	}
	return Result;
}

UNeuralNetworkModel* UNeuralNetworkModel::CreateModel(UObject* Parent, FString RuntimeName, UNNEModelData* ModelData)
{
	using namespace UE::NNECore;

	if (!ModelData)
	{
		UE_LOG(LogTemp, Error, TEXT("Invalid model data"));
		return nullptr;
	}

	TWeakInterfacePtr<INNERuntimeCPU> Runtime = GetRuntime<INNERuntimeCPU>(RuntimeName);
	if (!Runtime.IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("No CPU runtime '%s' found"), *RuntimeName);
		return nullptr;
	}

	TUniquePtr<IModelCPU> UniqueModel = Runtime->CreateModelCPU(ModelData);
	if (!UniqueModel.IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("Could not create the CPU model"));
		return nullptr;
	}

	UNeuralNetworkModel* Result = NewObject<UNeuralNetworkModel>(Parent);
	if (Result)
	{
		Result->Model = TSharedPtr<IModelCPU>(UniqueModel.Release());
		return Result;
	}

	return nullptr;
}

bool UNeuralNetworkModel::CreateTensor(TArray<int32> Shape, UPARAM(ref) FNeuralNetworkTensor& Tensor)
{
	if (Shape.Num() == 0)
	{
		return false;
	}

	int32 Volume = 1;
	for (int32 i = 0; i < Shape.Num(); i++)
	{
		if (Shape[i] < 1)
		{
			return false;
		}
		Volume *= Shape[i];
	}

	Tensor.Shape = Shape;
	Tensor.Data.SetNum(Volume);
	return true;
}

int32 UNeuralNetworkModel::NumInputs()
{
	check(Model.IsValid())
		return Model->GetInputTensorDescs().Num();
}

int32 UNeuralNetworkModel::NumOutputs()
{
	check(Model.IsValid())
		return Model->GetOutputTensorDescs().Num();
}

TArray<int32> UNeuralNetworkModel::GetInputShape(int32 Index)
{
	check(Model.IsValid())

		using namespace UE::NNECore;

	TConstArrayView<FTensorDesc> Desc = Model->GetInputTensorDescs();
	if (Index < 0 || Index >= Desc.Num())
	{
		return TArray<int32>();
	}

	return TArray<int32>(Desc[Index].GetShape().GetData());
}

TArray<int32> UNeuralNetworkModel::GetOutputShape(int32 Index)
{
	check(Model.IsValid())

		using namespace UE::NNECore;

	TConstArrayView<FTensorDesc> Desc = Model->GetOutputTensorDescs();
	if (Index < 0 || Index >= Desc.Num())
	{
		return TArray<int32>();
	}

	return TArray<int32>(Desc[Index].GetShape().GetData());
}

bool UNeuralNetworkModel::SetInputs(const TArray<FNeuralNetworkTensor>& Inputs)
{
	check(Model.IsValid())

		using namespace UE::NNECore;

	InputBindings.Reset();
	InputShapes.Reset();

	TConstArrayView<FTensorDesc> InputDescs = Model->GetInputTensorDescs();
	if (InputDescs.Num() != Inputs.Num())
	{
		UE_LOG(LogTemp, Error, TEXT("Invalid number of input tensors provided"));
		return false;
	}

	InputBindings.SetNum(Inputs.Num());
	InputShapes.SetNum(Inputs.Num());
	for (int32 i = 0; i < Inputs.Num(); i++)
	{
		InputBindings[i].Data = (void*)Inputs[i].Data.GetData();
		InputBindings[i].SizeInBytes = Inputs[i].Data.Num() * sizeof(float);
		InputShapes[i] = FTensorShape::MakeFromSymbolic(FSymbolicTensorShape::Make(Inputs[i].Shape));
	}

	if (Model->SetInputTensorShapes(InputShapes) != 0)
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to set the input shapes"));
		return false;
	}

	return true;
}

bool UNeuralNetworkModel::RunSync(UPARAM(ref) TArray<FNeuralNetworkTensor>& Outputs)
{
	check(Model.IsValid());

	using namespace UE::NNECore;

	TConstArrayView<FTensorDesc> OutputDescs = Model->GetOutputTensorDescs();
	if (OutputDescs.Num() != Outputs.Num())
	{
		UE_LOG(LogTemp, Error, TEXT("Invalid number of output tensors provided"));
		return false;
	}

	TArray<FTensorBindingCPU> OutputBindings;
	OutputBindings.SetNum(Outputs.Num());
	for (int32 i = 0; i < Outputs.Num(); i++)
	{
		OutputBindings[i].Data = (void*)Outputs[i].Data.GetData();
		OutputBindings[i].SizeInBytes = Outputs[i].Data.Num() * sizeof(float);
	}

	return Model->RunSync(InputBindings, OutputBindings) == 0;
}