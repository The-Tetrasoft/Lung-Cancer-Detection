import React, { ReactNode, useState } from 'react';
import { Button } from './ui/button';

interface TouchButtonProps {
  children: ReactNode;
  onClick?: () => void;
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
  size?: 'default' | 'sm' | 'lg' | 'icon';
  className?: string;
  disabled?: boolean;
  type?: 'button' | 'submit' | 'reset';
}

// Touch-optimized button with haptic feedback simulation
export function TouchButton({
  children,
  onClick,
  variant = 'default',
  size = 'default',
  className = '',
  disabled = false,
  type = 'button'
}: TouchButtonProps) {
  const [isPressed, setIsPressed] = useState(false);

  const handleTouchStart = () => {
    setIsPressed(true);
    // Simulate haptic feedback on supported devices
    if ('vibrate' in navigator) {
      navigator.vibrate(50);
    }
  };

  const handleTouchEnd = () => {
    setIsPressed(false);
  };

  const handleClick = () => {
    if (onClick && !disabled) {
      onClick();
    }
  };

  return (
    <Button
      type={type}
      variant={variant}
      size={size}
      disabled={disabled}
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
      onMouseDown={handleTouchStart}
      onMouseUp={handleTouchEnd}
      onClick={handleClick}
      className={`
        mobile-button touch-manipulation
        transform transition-transform duration-75
        ${isPressed ? 'scale-95' : 'scale-100'}
        ${className}
      `}
    >
      {children}
    </Button>
  );
}

// Touch-optimized input with better mobile UX
interface TouchInputProps {
  type?: string;
  placeholder?: string;
  value?: string;
  onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onFocus?: () => void;
  onBlur?: () => void;
  className?: string;
  disabled?: boolean;
  required?: boolean;
  autoComplete?: string;
}

export function TouchInput({
  type = 'text',
  placeholder,
  value,
  onChange,
  onFocus,
  onBlur,
  className = '',
  disabled = false,
  required = false,
  autoComplete
}: TouchInputProps) {
  return (
    <input
      type={type}
      placeholder={placeholder}
      value={value}
      onChange={onChange}
      onFocus={onFocus}
      onBlur={onBlur}
      disabled={disabled}
      required={required}
      autoComplete={autoComplete}
      className={`
        mobile-input touch-manipulation
        w-full border border-input bg-input-background
        rounded-md px-3 py-2 text-sm
        focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2
        disabled:cursor-not-allowed disabled:opacity-50
        ${className}
      `}
    />
  );
}

// Mobile-friendly toggle switch
interface TouchToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  disabled?: boolean;
  className?: string;
}

export function TouchToggle({
  checked,
  onChange,
  label,
  disabled = false,
  className = ''
}: TouchToggleProps) {
  const handleClick = () => {
    if (!disabled) {
      onChange(!checked);
      // Haptic feedback
      if ('vibrate' in navigator) {
        navigator.vibrate(30);
      }
    }
  };

  return (
    <div className={`flex items-center space-x-3 ${className}`}>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={handleClick}
        className={`
          relative inline-flex h-6 w-11 items-center rounded-full 
          transition-colors duration-200 ease-in-out
          focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2
          touch-manipulation min-h-[44px] min-w-[44px] p-4
          ${checked ? 'bg-blue-600' : 'bg-slate-200'}
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
        `}
      >
        <span
          aria-hidden="true"
          className={`
            inline-block h-4 w-4 transform rounded-full bg-white 
            transition-transform duration-200 ease-in-out
            ${checked ? 'translate-x-5' : 'translate-x-1'}
          `}
        />
      </button>
      {label && (
        <label
          className={`text-sm font-medium ${disabled ? 'opacity-50' : 'cursor-pointer'}`}
          onClick={!disabled ? handleClick : undefined}
        >
          {label}
        </label>
      )}
    </div>
  );
}

// Mobile-friendly modal overlay
interface TouchModalProps {
  isOpen: boolean;
  onClose: () => void;
  children: ReactNode;
  title?: string;
}

export function TouchModal({ isOpen, onClose, children, title }: TouchModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
           <div className="
             relative bg-background rounded-t-lg sm:rounded-lg border shadow-lg
             w-full max-w-lg md:max-w-xl mx-4 mb-0 sm:mb-4
             max-h-[90vh] overflow-y-auto
             mobile-safe-area
           ">
        {title && (
          <div className="flex items-center justify-between p-4 border-b">
            <h2 className="text-lg font-semibold">{title}</h2>
            <TouchButton
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="h-8 w-8 p-0"
            >
              ×
            </TouchButton>
          </div>
        )}
        <div className="p-4">
          {children}
        </div>
      </div>
    </div>
  );
}