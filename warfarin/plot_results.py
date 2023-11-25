from warfarin.metrics import Divergence, SupportCoverage


def main():
    print("Surrogate Cost:", preds[-1][0], preds[-1][1] / np.sqrt(len(X_test)))
    print("Oracle Cost:", gts[-1][0], gts[-1][1] / np.sqrt(len(X_test)))
    doses_gen = col_transforms[dataset.dose].invert(policy.optimum())
    doses_true = col_transforms[dataset.dose].invert(X_test[dataset.dose])
    print("Support Coverage:", SupportCoverage(doses_true, doses_gen))
    print("JS Divergence:", Divergence(doses_true, doses_gen))

    # Plot relevant metrics.
    preds = (
        np.array([mean for mean, _ in preds]),
        np.array([std for _, std in preds]) / np.sqrt(len(X_test))
    )
    gts = (
        np.array([mean for mean, _ in gts]),
        np.array([std for _, std in gts]) / np.sqrt(len(X_test))
    )
    plt.figure(figsize=(10, 5))
    steps = args.batch_size * np.arange(len(preds[0]))
    for (mean, sem), label in zip([preds, gts], ["Surrogate", "Oracle"]):
        plt.plot(steps, mean, label=label)
        plt.fill_between(steps, mean - sem, mean + sem, alpha=0.1)
    plt.xlabel("Optimization Steps")
    plt.ylabel("Warfarin-Associated Dosage Cost")
    plt.xlim(np.min(steps), np.max(steps))
    plt.ylim(0, 1000)
    plt.legend()
    if args.savepath is None:
        plt.show()
    else:
        plt.savefig(
            args.savepath, dpi=600, transparent=True, bbox_inches="tight"
        )
    plt.close()
    return


if __name__ == "__main__":
    main()
