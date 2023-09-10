# Initialize avg-level metrics for validation
avg_test_dice_score = 0
avg_test_iou = 0
avg_test_sensitivity = 0
avg_test_specificity = 0
avg_test_accuracy = 0

with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_criterion(output, target)

        # calculate_and_log_metrics(output, target, loss, writer, test_steps, None, None, None, 'test')
        metrics = calculate_and_log_metrics(output, target, loss, writer, test_steps, None, None, None, 'test',
                                            do_visualize)

        # Update avg-level metrics
        avg_test_dice_score += metrics['dice_score']
        avg_test_iou += metrics['iou']
        avg_test_sensitivity += metrics['sensitivity']
        avg_test_specificity += metrics['specificity']
        avg_test_accuracy += metrics['accuracy']

        test_steps += 1  # Test 결과 저장

# Calculate average avg-level metrics for testidation
num_batches_test = len(test_loader)
avg_test_dice_score /= num_batches_test
avg_test_iou /= num_batches_test
avg_test_sensitivity /= num_batches_test
avg_test_specificity /= num_batches_test
avg_test_accuracy /= num_batches_test

# Log average avg-level metrics to TensorBoard
writer.add_scalar('avg_test_Dice_Score', avg_test_dice_score, 0)
writer.add_scalar('avg_test_IoU', avg_test_iou, 0)
writer.add_scalar('avg_test_Sensitivity', avg_test_sensitivity, 0)
writer.add_scalar('avg_test_Specificity', avg_test_specificity, 0)
writer.add_scalar('avg_test_Accuracy', avg_test_accuracy, 0)