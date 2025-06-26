#!/usr/bin/env python3
"""
monitor_training.py - Monitor training progress in real-time
"""
import os
import sys
import time
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import re

def parse_log_file(log_path):
    """Parse training log file for metrics"""
    epochs = []
    train_loss = []
    train_dice = []
    val_loss = []
    val_dice = []
    learning_rates = []
    
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Look for epoch summary lines
        if "Epoch" in line and "Summary:" in line:
            # Extract epoch number
            epoch_match = re.search(r'Epoch (\d+)', line)
            if epoch_match:
                epochs.append(int(epoch_match.group(1)))
        
        # Extract metrics
        if "Loss:" in line:
            loss_match = re.search(r'Loss: ([\d.]+)', line)
            if loss_match:
                train_loss.append(float(loss_match.group(1)))
        
        if "Dice:" in line and "Val Dice:" not in line:
            dice_match = re.search(r'Dice: ([\d.]+)', line)
            if dice_match:
                train_dice.append(float(dice_match.group(1)))
        
        if "Val Dice:" in line:
            val_dice_match = re.search(r'Val Dice: ([\d.]+)', line)
            if val_dice_match:
                val_dice.append(float(val_dice_match.group(1)))
        
        # Learning rate changes
        if "new learning rate" in line.lower():
            lr_match = re.search(r'([\d.e-]+)', line)
            if lr_match:
                learning_rates.append((len(epochs), float(lr_match.group(1))))
    
    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'train_dice': train_dice,
        'val_dice': val_dice,
        'learning_rates': learning_rates
    }

def plot_training_progress(metrics, save_path=None):
    """Create training progress plots"""
    if not metrics or not metrics['epochs']:
        print("No data to plot yet...")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training Loss
    ax1.plot(metrics['epochs'][:len(metrics['train_loss'])], 
             metrics['train_loss'], 'b-', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Dice Coefficient
    if metrics['train_dice']:
        ax2.plot(metrics['epochs'][:len(metrics['train_dice'])], 
                 metrics['train_dice'], 'g-', label='Train Dice')
    if metrics['val_dice']:
        ax2.plot(metrics['epochs'][:len(metrics['val_dice'])], 
                 metrics['val_dice'], 'r-', label='Val Dice')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Coefficient')
    ax2.set_title('Dice Score Progress')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Learning Rate
    if metrics['learning_rates']:
        lr_epochs, lr_values = zip(*metrics['learning_rates'])
        ax3.semilogy(lr_epochs, lr_values, 'mo-')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.grid(True, alpha=0.3)
    
    # Summary Stats
    ax4.axis('off')
    summary_text = f"Training Summary\n" + "="*30 + "\n"
    if metrics['epochs']:
        summary_text += f"Current Epoch: {metrics['epochs'][-1]}\n"
    if metrics['train_dice']:
        summary_text += f"Best Train Dice: {max(metrics['train_dice']):.4f}\n"
    if metrics['val_dice']:
        summary_text += f"Best Val Dice: {max(metrics['val_dice']):.4f}\n"
        best_epoch = metrics['epochs'][metrics['val_dice'].index(max(metrics['val_dice']))]
        summary_text += f"Best Val Epoch: {best_epoch}\n"
    if metrics['train_loss']:
        summary_text += f"Current Loss: {metrics['train_loss'][-1]:.4f}\n"
    
    ax4.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

def monitor_loop(log_path, interval=30, plot_dir=None):
    """Monitor training progress in a loop"""
    print(f"Monitoring: {log_path}")
    print(f"Update interval: {interval} seconds")
    print("Press Ctrl+C to stop\n")
    
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    
    while True:
        try:
            # Parse log file
            metrics = parse_log_file(log_path)
            
            if metrics:
                # Clear screen (works on Unix-like systems)
                os.system('clear')
                
                # Print summary
                print(f"Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*50)
                
                if metrics['epochs']:
                    print(f"Current Epoch: {metrics['epochs'][-1]}")
                if metrics['train_dice']:
                    print(f"Current Train Dice: {metrics['train_dice'][-1]:.4f}")
                if metrics['val_dice']:
                    print(f"Current Val Dice: {metrics['val_dice'][-1]:.4f}")
                    print(f"Best Val Dice: {max(metrics['val_dice']):.4f}")
                
                # Save plot
                if plot_dir:
                    plot_path = os.path.join(plot_dir, 'training_progress.png')
                    plot_training_progress(metrics, plot_path)
            else:
                print("Waiting for training to start...")
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval)

def main():
    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument('log_path', help='Path to training log file')
    parser.add_argument('--interval', type=int, default=30,
                        help='Update interval in seconds')
    parser.add_argument('--plot-dir', type=str, default=None,
                        help='Directory to save progress plots')
    parser.add_argument('--once', action='store_true',
                        help='Run once instead of continuous monitoring')
    
    args = parser.parse_args()
    
    if args.once:
        metrics = parse_log_file(args.log_path)
        if metrics:
            plot_training_progress(metrics, 
                                 os.path.join(args.plot_dir, 'training_progress.png') 
                                 if args.plot_dir else None)
    else:
        monitor_loop(args.log_path, args.interval, args.plot_dir)

if __name__ == '__main__':
    main()
