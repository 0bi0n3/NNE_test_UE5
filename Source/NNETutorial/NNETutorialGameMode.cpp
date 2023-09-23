// Copyright Epic Games, Inc. All Rights Reserved.

#include "NNETutorialGameMode.h"
#include "NNETutorialCharacter.h"
#include "UObject/ConstructorHelpers.h"

ANNETutorialGameMode::ANNETutorialGameMode()
{
	// set default pawn class to our Blueprinted character
	static ConstructorHelpers::FClassFinder<APawn> PlayerPawnBPClass(TEXT("/Game/ThirdPerson/Blueprints/BP_ThirdPersonCharacter"));
	if (PlayerPawnBPClass.Class != NULL)
	{
		DefaultPawnClass = PlayerPawnBPClass.Class;
	}
}
