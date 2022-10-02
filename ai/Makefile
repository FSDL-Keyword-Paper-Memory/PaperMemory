get-fresh-data:
	sh tasks/get-fresh-data.sh

clean-abstracts: $(path)
	sh tasks/clean-abstracts.sh $(path)

prepare-devset:
	sh tasks/prepare-devset.sh $(path)

validate:
	sh tasks/validate.sh $(path)

predict:
	sh tasks/predict.sh $(model) $(threshold) "$(abstract)"
