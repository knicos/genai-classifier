import { Spinner } from '@base/main';
import { Story, StoryDefault } from '@ladle/react';
import { Theme } from './decorators';
import './style.css';

export default {
    decorators: [Theme],
} satisfies StoryDefault;

export const Small: Story = () => <Spinner size="small" />;
export const Large: Story = () => <Spinner size="large" />;
export const Disabled: Story = () => (
    <Spinner
        size="small"
        disabled
    />
);
