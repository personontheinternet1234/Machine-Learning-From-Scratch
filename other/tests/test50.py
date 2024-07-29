from gardenpy.utils import ansi

save_parameters = input(f"Input '{ansi['white']}{ansi['italic']}save{ansi['reset']}' to save parameters: ")
if save_parameters.lower() == 'save':
    print('not done yet :(')
else:
    print(f"{ansi['bright_black']}{ansi['italic']}Parameters not saved.{ansi['reset']}")