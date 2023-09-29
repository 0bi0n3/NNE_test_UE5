// Fill out your copyright notice in the Description page of Project Settings.


#include "AMyNeuralNetworkActor.h"
#include "NeuralNetworkModel.h"

// Sets default values
AAMyNeuralNetworkActor::AAMyNeuralNetworkActor()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void AAMyNeuralNetworkActor::BeginPlay()
{
	Super::BeginPlay();

	NeuralNetworkModel = UNeuralNetworkModel::CreateModel(this, TEXT("NNERuntimeORTCpu"), ModelData);

}

// Called every frame
void AAMyNeuralNetworkActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime); 
	
	// Run the model.
	TArray<FNeuralNetworkTensor> Outputs;
	// Assume Outputs is properly set up.
	bool bSuccess = NeuralNetworkModel->RunSync(Outputs);
	if (bSuccess)
	{
		// Access the output data.
		for (int32 i = 0; i < Outputs.Num(); ++i)
		{
			float OutputValue = Outputs[i].Data[0];

			// Print output value to screen.
			GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red, FString::Printf(TEXT("Model output %d: %f"), i, OutputValue));
		}
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to run the model"));
	}

}

