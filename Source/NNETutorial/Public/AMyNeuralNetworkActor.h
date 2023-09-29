// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "NeuralNetworkModel.h"
#include "AMyNeuralNetworkActor.generated.h"

UCLASS()
class NNETUTORIAL_API AAMyNeuralNetworkActor : public AActor
{
	GENERATED_BODY()
	
public:
	UPROPERTY(EditAnywhere)
	UNeuralNetworkModel* NeuralNetworkModel;

	UPROPERTY(EditAnywhere, Category = "Neural Network")
	UNNEModelData* ModelData;
	// Sets default values for this actor's properties
	AAMyNeuralNetworkActor();

	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	// Called every frame
	virtual void Tick(float DeltaTime) override;

};
