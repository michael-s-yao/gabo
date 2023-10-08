import torch


def main():
    model = "./ckpts/warfarin_cost_estimator.ckpt"
    model = torch.load(model, map_location=torch.device("cpu"))
    torch.save(model, "./ckpts/warfarin_cost_estimator_cpu.ckpt")
    # print(model)


if __name__ == "__main__":
    main()
